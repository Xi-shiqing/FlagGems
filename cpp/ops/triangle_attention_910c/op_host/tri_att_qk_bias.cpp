#include "tri_att_qk_bias_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_M = 64;
static constexpr uint32_t BLOCK_N = 64;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &scoreShape = context->GetInputShape(0)->GetOriginShape();
    const uint32_t m = static_cast<uint32_t>(scoreShape.GetDim(0));
    const uint32_t n = static_cast<uint32_t>(scoreShape.GetDim(1));

    TriAttQkBiasTilingData tiling;
    tiling.set_m(m);
    tiling.set_n(n);
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
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *scoreShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, scoreShape->GetDim(0));
    outShape->SetDim(1, scoreShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class TriAttQkBias : public OpDef {
public:
    explicit TriAttQkBias(const char *name) : OpDef(name)
    {
        this->Input("score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
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

OP_ADD(TriAttQkBias);
}  // namespace ops
