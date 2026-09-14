#include "tri_att_fa_official_bias_probe_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/matmul/bmm_tiling.h"

#ifndef TRIATT_MAX_KV_STACK_LEN
#define TRIATT_MAX_KV_STACK_LEN 384
#endif

namespace optiling {
static bool BuildBatchMatmulTiling(bool transposeB, uint32_t aHeads, uint32_t bHeads, uint32_t cHeads,
                                   uint32_t m, uint32_t n, uint32_t k, TCubeTiling &out)
{
    matmul_tiling::BatchMatmulTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16, transposeB);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(m, n, k);
    mm.SetOrgShape(m, n, k);
    if (mm.SetALayout(1, m, aHeads, 1, k) != 0) {
        return false;
    }
    if (transposeB) {
        if (mm.SetBLayout(1, n, bHeads, 1, k) != 0) {
            return false;
        }
    } else if (mm.SetBLayout(1, k, bHeads, 1, n) != 0) {
        return false;
    }
    if (mm.SetCLayout(1, m, cHeads, 1, n) != 0 || mm.SetBatchNum(1) != 0) {
        return false;
    }
    return mm.GetTiling(out) != -1;
}
static constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 384ULL * TRIATT_MAX_KV_STACK_LEN;
static constexpr uint64_t PRE_LAUNCH = 2ULL;
static constexpr uint32_t DEFAULT_qTileSize = 192;
static constexpr uint32_t KV_STACK_SIZE = 384;
static constexpr uint32_t BMM_HEAD_BATCH = 1;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &qShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &kShape = context->GetInputShape(1)->GetOriginShape();
    const gert::Shape &bias1Shape = context->GetInputShape(3)->GetOriginShape();
    const gert::Shape &bias2Shape = context->GetInputShape(4)->GetOriginShape();
    const gert::Shape &actualQShape = context->GetInputShape(5)->GetOriginShape();
    const uint32_t batch = static_cast<uint32_t>(actualQShape.GetDim(0));
    const uint32_t totalM = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t totalN = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t m = totalM / batch;
    const uint32_t qHeads = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t n = totalN / batch;
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t qTileSize = (qHeads <= 2) ? 128 : ((qHeads <= 4) ? 192 : 256);
    const uint32_t bias1Stride = static_cast<uint32_t>(bias1Shape.GetDim(1));
    // Each TND batch has one Bias1 row that is broadcast over its query rows.
    const uint32_t bias1Rows = 1;
    const uint32_t bias2Stride = static_cast<uint32_t>(bias2Shape.GetDim(bias2Shape.GetDimNum() - 1));
    // Protenix Bias2 is shared over the flattened N groups when its leading dimension is 1.
    const uint32_t bias2BatchShared =
        (bias2Shape.GetDimNum() == 3 || bias2Shape.GetDim(0) == 1) ? 1 : 0;
    const uint32_t bias2BatchCount =
        (!bias2BatchShared && bias2Shape.GetDimNum() >= 5) ?
        static_cast<uint32_t>(bias2Shape.GetDim(0)) : batch;
    const uint32_t bias2GroupsPerBatch =
        (bias2BatchCount == 0 || batch % bias2BatchCount != 0) ?
        1 : batch / bias2BatchCount;
    const uint32_t qTiles = (m + qTileSize - 1) / qTileSize;
    const uint32_t kvTiles = (n + 127) / 128;
    const uint32_t qHeadBlocks = (qHeads + BMM_HEAD_BATCH - 1) / BMM_HEAD_BATCH;
    const uint32_t qTail = (m % qTileSize == 0) ? qTileSize : (m % qTileSize);
    const uint32_t kvTail = (n % KV_STACK_SIZE == 0) ? KV_STACK_SIZE : (n % KV_STACK_SIZE);
#ifdef TRIATT_BMM_FIXED_BLOCK_DIM
    const uint32_t blockDim = TRIATT_BMM_FIXED_BLOCK_DIM;
#else
    const uint32_t taskCount = qTiles * qHeadBlocks * batch;
    const uint32_t blockDim = taskCount < 24 ? taskCount : 24;
#endif

    TriAttFaOfficialBiasProbeTilingData tiling;
    const uint32_t activeBlocks = blockDim == 0 ? 1 : blockDim;
    const uint64_t perCore = static_cast<uint64_t>(WORKSPACE_BLOCK_SIZE_DB) * (PRE_LAUNCH + 1);
    tiling.set_mm1OutSize(perCore * activeBlocks * sizeof(float));
    tiling.set_smOnlineOutSize(perCore * activeBlocks * 2);
    tiling.set_mm2OutSize(perCore * activeBlocks * sizeof(float));
    tiling.set_batch(batch);
    tiling.set_numHeads(qHeads);
    tiling.set_kvHeads(qHeads);
    tiling.set_embeddingSize(d);
    tiling.set_embeddingSizeV(d);
    tiling.set_blockSize(128);
    tiling.set_maxNumBlocksPerBatch(kvTiles);
    tiling.set_firstBatchTaskNum(qTiles * qHeadBlocks);
    tiling.set_totalTaskNum(qTiles * qHeadBlocks * batch);
    tiling.set_maskType(0);
    tiling.set_scaleValue(1.0f);
    tiling.set_pseStride(bias2Stride);
    tiling.set_bias1Stride(bias1Stride);
    tiling.set_bias1Rows(bias1Rows);
    tiling.set_bias2BatchShared(bias2BatchShared);
    tiling.set_bias2GroupsPerBatch(bias2GroupsPerBatch);
    tiling.set_qHeadStride(m * d);
    tiling.set_kvHeadStride(n * d);
    tiling.set_qTileSize(qTileSize);
    tiling.set_kvStackSize(KV_STACK_SIZE);
    if (!BuildBatchMatmulTiling(true, qHeads, qHeads, 1, qTileSize, KV_STACK_SIZE, d, tiling.mm1FullFull) ||
        !BuildBatchMatmulTiling(true, qHeads, qHeads, 1, qTileSize, kvTail, d, tiling.mm1FullTail) ||
        !BuildBatchMatmulTiling(true, qHeads, qHeads, 1, qTail, KV_STACK_SIZE, d, tiling.mm1TailFull) ||
        !BuildBatchMatmulTiling(true, qHeads, qHeads, 1, qTail, kvTail, d, tiling.mm1TailTail) ||
        !BuildBatchMatmulTiling(false, 1, qHeads, 1, qTileSize, d, KV_STACK_SIZE, tiling.mm2FullFull) ||
        !BuildBatchMatmulTiling(false, 1, qHeads, 1, qTileSize, d, kvTail, tiling.mm2FullTail) ||
        !BuildBatchMatmulTiling(false, 1, qHeads, 1, qTail, d, KV_STACK_SIZE, tiling.mm2TailFull) ||
        !BuildBatchMatmulTiling(false, 1, qHeads, 1, qTail, d, kvTail, tiling.mm2TailTail)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    const uint64_t workspaceBytesPerCore = perCore * (sizeof(float) + 2 + sizeof(float) + sizeof(float));
    const uint64_t workspaceBytes = workspaceBytesPerCore * activeBlocks;
    constexpr uint64_t RESERVED_WORKSPACE = 16ULL * 1024ULL * 1024ULL;
    currentWorkspace[0] = static_cast<size_t>(workspaceBytes + tiling.get_mm1OutSize() + RESERVED_WORKSPACE);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(qShape->GetDimNum());
    for (size_t i = 0; i < qShape->GetDimNum(); ++i) {
        outShape->SetDim(i, qShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // Q/K/V and output use the same dtype; runtime input selects FP16 or FP32.
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TriAttFaOfficialBiasProbe : public OpDef {
public:
    explicit TriAttFaOfficialBiasProbe(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias1").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias2").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_q").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_kv").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
    }
};
OP_ADD(TriAttFaOfficialBiasProbe);
}
