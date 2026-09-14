#include "tri_att_fa_official_probe_tiling.h"
#include "register/op_def_registry.h"

#ifndef TRIATT_MAX_KV_STACK_LEN
#define TRIATT_MAX_KV_STACK_LEN 384
#endif

namespace optiling {
static constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 128ULL * TRIATT_MAX_KV_STACK_LEN;
static constexpr uint64_t PRE_LAUNCH = 2ULL;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &qShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &kShape = context->GetInputShape(1)->GetOriginShape();
    const gert::Shape &pseShape = context->GetInputShape(3)->GetOriginShape();
    const uint32_t m = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t qHeads = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t n = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t pseStride = static_cast<uint32_t>(pseShape.GetDim(2));
    const uint32_t qTiles = (m + 127) / 128;
    const uint32_t kvTiles = (n + 127) / 128;
#ifdef TRIATT_FIXED_BLOCK_DIM
    const uint32_t blockDim = TRIATT_FIXED_BLOCK_DIM;
#else
    const uint32_t blockDim = qTiles * qHeads;
#endif

    TriAttFaOfficialProbeTilingData tiling;
    const uint32_t activeBlocks = blockDim == 0 ? 1 : blockDim;
    const uint64_t perCore = static_cast<uint64_t>(WORKSPACE_BLOCK_SIZE_DB) * (PRE_LAUNCH + 1);
    tiling.set_mm1OutSize(perCore * activeBlocks * sizeof(float));
    tiling.set_smOnlineOutSize(perCore * activeBlocks * 2);
    tiling.set_mm2OutSize(perCore * activeBlocks * sizeof(float));
    tiling.set_batch(1);
    tiling.set_numHeads(qHeads);
    tiling.set_kvHeads(qHeads);
    tiling.set_embeddingSize(d);
    tiling.set_embeddingSizeV(d);
    tiling.set_blockSize(128);
    tiling.set_maxNumBlocksPerBatch(kvTiles);
    tiling.set_firstBatchTaskNum(qTiles * qHeads);
    tiling.set_totalTaskNum(qTiles * qHeads);
    tiling.set_maskType(0);
    tiling.set_scaleValue(1.0f);
    tiling.set_pseStride(pseStride);
    tiling.set_bias1Stride(0);
    tiling.set_bias1Rows(0);
    tiling.set_bias2BatchShared(0);
    tiling.set_bias2GroupsPerBatch(1);

    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    const uint64_t workspaceBytesPerCore = perCore * (sizeof(float) + 2 + sizeof(float) + sizeof(float));
    const uint64_t workspaceBytes = workspaceBytesPerCore * activeBlocks;
    constexpr uint64_t RESERVED_WORKSPACE = 16ULL * 1024ULL * 1024ULL;
    currentWorkspace[0] = static_cast<size_t>(workspaceBytes + RESERVED_WORKSPACE);
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
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TriAttFaOfficialProbe : public OpDef {
public:
    explicit TriAttFaOfficialProbe(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pse").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("actual_q").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("actual_kv").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
    }
};
OP_ADD(TriAttFaOfficialProbe);
}
