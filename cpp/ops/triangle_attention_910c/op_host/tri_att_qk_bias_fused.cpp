#include "tri_att_qk_bias_fused_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = 64;
static constexpr uint32_t BLOCK_K = 32;
static constexpr size_t RESERVED_WORKSPACE_BYTES = 16 * 1024 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &qShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &kShape = context->GetInputShape(1)->GetOriginShape();
    const uint32_t m = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t n = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t k = static_cast<uint32_t>(qShape.GetDim(1));

    TriAttQkBiasFusedTilingData tiling;
    tiling.set_m(m);
    tiling.set_n(n);
    tiling.set_k(k);
    tiling.set_blockM(BLOCK_M);
    tiling.set_blockN(BLOCK_N);

    const uint32_t tilesM = (m + BLOCK_M - 1) / BLOCK_M;
    const uint32_t tilesN = (n + BLOCK_N - 1) / BLOCK_N;
    const uint32_t blockDim = tilesM * tilesN;
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    const size_t workspaceBytes = static_cast<size_t>(m) * n * sizeof(float);
    currentWorkspace[0] = RESERVED_WORKSPACE_BYTES + ((workspaceBytes + 31) / 32) * 32;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(0);
    const gert::Shape *kShape = context->GetInputShape(1);
    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, qShape->GetDim(0));
    outShape->SetDim(1, kShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class TriAttQkBiasFused : public OpDef {
public:
    explicit TriAttQkBiasFused(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93");
    }
};

OP_ADD(TriAttQkBiasFused);
}  // namespace ops
