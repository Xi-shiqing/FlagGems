#include "kernel_operator.h"

namespace AscendC { __aicore__ inline void clearWorkspace(__gm__ uint8_t *) {} }
namespace matmul { __aicore__ inline void clearWorkspace(__gm__ uint8_t *) {} }

struct FAInferTilingData {
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    float scaleValue;
    uint32_t pseStride;
    uint32_t bias1Stride;
    uint32_t bias1Rows;
    uint32_t bias2BatchShared;
    uint32_t bias2GroupsPerBatch;
};

#include "../official_fa/flash_attention_regular.h"

using namespace NpuArch;
using namespace KernelCommon;

namespace SplitFuseProbe {
using ArchTag = Arch::AtlasA2;
using ElementQ = half;
using LayoutQ = layout::RowMajor;
using ElementK = half;
using LayoutK = layout::ColumnMajor;
using ElementV = half;
using LayoutV = layout::RowMajor;
using ElementS = float;
using LayoutS = layout::RowMajor;
using ElementP = half;
using LayoutP = layout::RowMajor;
using ElementO = half;
using LayoutO = layout::RowMajor;
using ElementLse = float;
using LayoutLse = layout::RowMajor;
using ElementMask = int8_t;
using LayoutMask = layout::RowMajor;
using ElementOTmp = float;
using LayoutOTmp = layout::RowMajor;
using ElementUpdate = float;
using LayoutUpdate = layout::RowMajor;

using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
using L0TileShapeQK = GemmShape<128, 128, 128>;
using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<false, false>;
using QType = Gemm::GemmType<ElementQ, LayoutQ>;
using KType = Gemm::GemmType<ElementK, LayoutK>;
using SType = Gemm::GemmType<ElementS, LayoutS>;
using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::NONE, float>;
using PType = Gemm::GemmType<ElementP, LayoutP>;
using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

using L1TileShapePV = GemmShape<128, 128, 256>;
using L0TileShapePV = GemmShape<128, 128, 128>;
using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<false, false>;
using VType = Gemm::GemmType<ElementV, LayoutV>;
using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, float>;
using OType = Gemm::GemmType<ElementO, LayoutO>;
using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

using DispatchPolicyInitOut = Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
using EpilogueInitOut = Epilogue::Block::BlockEpilogue<DispatchPolicyInitOut, OType, LseType>;
using Kernel = SplitFuse::FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>;
}

extern "C" __global__ __aicore__ void tri_att_fa_official_probe(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR pse, GM_ADDR actual_q, GM_ADDR actual_kv,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(FAInferTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KernelCommon::FAIKernelParams params{
        query, key, value, nullptr, nullptr, actual_q, actual_kv,
        output, nullptr, user, tiling, pse};
    SplitFuseProbe::Kernel kernel;
    kernel(params);
}

#ifndef ASCENDC_CPU_DEBUG
void tri_att_fa_official_probe_do(uint32_t blockDim, void *l2ctrl, void *stream,
                                  uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *pse,
                                  uint8_t *actual_q, uint8_t *actual_kv, uint8_t *output,
                                  uint8_t *workspace, uint8_t *tiling)
{
    tri_att_fa_official_probe<<<blockDim, l2ctrl, stream>>>(query, key, value, pse, actual_q, actual_kv, output, workspace, tiling);
}
#endif
