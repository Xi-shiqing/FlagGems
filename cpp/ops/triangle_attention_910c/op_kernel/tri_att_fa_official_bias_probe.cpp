#include "kernel_operator.h"
#include "lib/matmul_intf.h"


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
    uint32_t qHeadStride;
    uint32_t kvHeadStride;
    uint32_t qTileSize;
    uint32_t kvStackSize;
    AscendC::tiling::TCubeTiling mm1FullFull;
    AscendC::tiling::TCubeTiling mm1FullTail;
    AscendC::tiling::TCubeTiling mm1TailFull;
    AscendC::tiling::TCubeTiling mm1TailTail;
    AscendC::tiling::TCubeTiling mm2FullFull;
    AscendC::tiling::TCubeTiling mm2FullTail;
    AscendC::tiling::TCubeTiling mm2TailFull;
    AscendC::tiling::TCubeTiling mm2TailTail;
};

#define TRIATT_MAX_KV_STACK_LEN 384
#define TRIATT_BMM 1
#include "../official_fa/flash_attention_regular.h"

using namespace NpuArch;
using namespace KernelCommon;

namespace TriAttBatchMmad {
using namespace AscendC;
using namespace AscendC::tiling;

constexpr uint32_t BATCH_HEADS = 1;
#ifdef DTYPE_QUERY
using TriAttQueryElement = DTYPE_QUERY;
using TriAttKeyElement = DTYPE_KEY;
using TriAttValueElement = DTYPE_VALUE;
#else
using TriAttQueryElement = half;
using TriAttKeyElement = half;
using TriAttValueElement = half;
#endif
using ATypeQK = MatmulType<TPosition::GM, CubeFormat::ND, TriAttQueryElement, false, LayoutMode::BSNGD>;
using BTypeQK = MatmulType<TPosition::GM, CubeFormat::ND, TriAttKeyElement, true, LayoutMode::BSNGD>;
using CTypeQK = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
using BiasTypeQK = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
using BiasTypePV = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BSNGD>;
using ATypePV = MatmulType<TPosition::GM, CubeFormat::ND, TriAttQueryElement, false, LayoutMode::BNGS1S2>;
using BTypePV = MatmulType<TPosition::GM, CubeFormat::ND, TriAttValueElement, false, LayoutMode::BSNGD>;
using CTypePV = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
constexpr MatmulConfig BMM_CONFIG = GetNormalConfig(true);
using MatmulQK = matmul::Matmul<ATypeQK, BTypeQK, CTypeQK, BiasTypeQK, BMM_CONFIG>;
using MatmulPV = matmul::Matmul<ATypePV, BTypePV, CTypePV, BiasTypePV, BMM_CONFIG>;

template <class MatmulT, class AType, class BType, class CType,
          class LayoutA_, class LayoutB_, class LayoutC_,
          bool TRANSPOSE_B, bool MINOR_IS_K, uint32_t BATCH_A, uint32_t BATCH_B>
class BatchedBlockMmad {
public:
    using ArchTag = Arch::AtlasA2;
    using L1TileShape = GemmShape<384, 128, 128>;
    using ElementA = typename AType::T;
    using LayoutA = LayoutA_;
    using ElementB = typename BType::T;
    using LayoutB = LayoutB_;
    using ElementC = typename CType::T;
    using LayoutC = LayoutC_;

    __aicore__ inline BatchedBlockMmad(Arch::Resource<ArchTag> &, uint32_t, uint32_t, uint32_t = 0) {}

    __aicore__ inline void SetTiling(
        const __gm__ AscendC::tiling::TCubeTiling *fullFull,
        const __gm__ AscendC::tiling::TCubeTiling *fullTail,
        const __gm__ AscendC::tiling::TCubeTiling *tailFull,
        const __gm__ AscendC::tiling::TCubeTiling *tailTail,
        uint32_t qTileSize, uint32_t kvStackSize)
    {
        tilingFullFull_ = fullFull;
        tilingFullTail_ = fullTail;
        tilingTailFull_ = tailFull;
        tilingTailTail_ = tailTail;
        qTileSize_ = qTileSize;
        kvStackSize_ = kvStackSize;
    }

    __aicore__ inline void SetRuntimeStrides(uint32_t strideA, uint32_t strideB, uint32_t strideC)
    {
        (void)strideA; (void)strideB; (void)strideC; strideA_ = 0; strideB_ = 0; strideC_ = 0;
    }

    __aicore__ inline void loadQGM(AscendC::GlobalTensor<ElementA>, LayoutA,
                                   uint32_t, uint32_t, uint32_t) {}

    template <typename... Args>
    __aicore__ inline void operator()(
        AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
        AscendC::GlobalTensor<ElementC> gC, AscendC::GlobalTensor<int32_t>,
        LayoutA, LayoutB, LayoutC, GemmCoord actualShape,
        Args...)
    {
#ifdef TRIATT_BMM
        const int actualM = static_cast<int>(actualShape.m() / BATCH_A);
        const int actualN = static_cast<int>(actualShape.n() / BATCH_B);
        const int actualK = static_cast<int>(actualShape.k());
        const bool minorTail = (MINOR_IS_K ? actualK : actualN) != static_cast<int>(kvStackSize_);
        MatmulT &activeMm = minorTail ? mmTail_ : mm_;
        if constexpr (MINOR_IS_K) {
            activeMm.SetOrgShape(actualM, actualN, actualK, actualN, actualN);
        } else {
            activeMm.SetOrgShape(actualM, actualK, actualK, actualK, actualN);
        }
        activeMm.SetTensorA(gA);
        if constexpr (TRANSPOSE_B) {
            activeMm.SetTensorB(gB, true);
        } else {
            activeMm.SetTensorB(gB);
        }
        activeMm.SetTail(actualM, actualN, actualK);
        activeMm.template IterateBatch<false, true>(
            gC, BATCH_A, BATCH_B, false, strideA_, strideB_, strideC_);
        activeMm.WaitIterateBatch();
#ifdef __DAV_C220_VEC__
        if constexpr (MINOR_IS_K) {
            AscendC::DataCacheCleanAndInvalid<ElementC, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_ALL>(gC);
        }
#endif
        activeMm.End();
#else
        const uint32_t m = actualShape.m() / BATCH_A;
        const uint32_t minor = MINOR_IS_K ? actualShape.k() : actualShape.n();
        const bool mTail = m != qTileSize_;
        const bool minorTail = minor != kvStackSize_;
        const __gm__ AscendC::tiling::TCubeTiling *selected =
            mTail ? (minorTail ? tilingTailTail_ : tilingTailFull_) :
                     (minorTail ? tilingFullTail_ : tilingFullFull_);
        AscendC::tiling::TCubeTiling selectedLocal;
        const __gm__ int32_t *selectedWords = reinterpret_cast<const __gm__ int32_t *>(selected);
        int32_t *localWords = reinterpret_cast<int32_t *>(&selectedLocal);
        for (uint32_t i = 0; i < sizeof(selectedLocal) / sizeof(int32_t); ++i) {
            localWords[i] = selectedWords[i];
        }
        mm_.SetSubBlockIdx(0);
        mm_.Init(&selectedLocal, GetTPipePtr());
        mm_.SetTensorA(gA);
        if constexpr (TRANSPOSE_B) {
            mm_.SetTensorB(gB, true);
        } else {
            mm_.SetTensorB(gB);
        }
        mm_.SetBatchNum(BATCH_A, BATCH_B);
        mm_.IterateBatch(gC, false, 0, false, strideA_, strideB_, strideC_);
        mm_.End();
#endif
    }

    __aicore__ inline auto &MatmulObject()
    {
        return mm_;
    }

    __aicore__ inline auto &MatmulObjectTail()
    {
        return mmTail_;
    }

private:
    MatmulT mm_;
    MatmulT mmTail_;
    const __gm__ AscendC::tiling::TCubeTiling *tilingFullFull_ = nullptr;
    const __gm__ AscendC::tiling::TCubeTiling *tilingFullTail_ = nullptr;
    const __gm__ AscendC::tiling::TCubeTiling *tilingTailFull_ = nullptr;
    const __gm__ AscendC::tiling::TCubeTiling *tilingTailTail_ = nullptr;
    uint32_t qTileSize_ = 192;
    uint32_t kvStackSize_ = 384;
    uint32_t strideA_ = 0;
    uint32_t strideB_ = 0;
    uint32_t strideC_ = 0;
};

using BlockMmadQK = BatchedBlockMmad<MatmulQK, ATypeQK, BTypeQK, CTypeQK,
                                    layout::RowMajor, layout::ColumnMajor, layout::RowMajor, true, false, 1, 1>;
using BlockMmadPV = BatchedBlockMmad<MatmulPV, ATypePV, BTypePV, CTypePV,
                                    layout::RowMajor, layout::RowMajor, layout::RowMajor, false, true, 1, 1>;
}

namespace SplitFuseProbe {
using ArchTag = Arch::AtlasA2;
#ifdef DTYPE_QUERY
// DynamicCompile supplies DTYPE_* for each runtime dtype signature.
using ElementQ = DTYPE_QUERY;
using LayoutQ = layout::RowMajor;
using ElementK = DTYPE_KEY;
using LayoutK = layout::ColumnMajor;
using ElementV = DTYPE_VALUE;
using LayoutV = layout::RowMajor;
using ElementP = DTYPE_QUERY;
using LayoutP = layout::RowMajor;
using ElementO = DTYPE_OUTPUT;
using LayoutO = layout::RowMajor;
#else
#ifdef TRIATT_FP32_QKV
using ElementQ = float;
using LayoutQ = layout::RowMajor;
using ElementK = float;
using LayoutK = layout::ColumnMajor;
using ElementV = float;
using LayoutV = layout::RowMajor;
using ElementP = float;
using LayoutP = layout::RowMajor;
using ElementO = float;
using LayoutO = layout::RowMajor;
#else
using ElementQ = half;
using LayoutQ = layout::RowMajor;
using ElementK = half;
using LayoutK = layout::ColumnMajor;
using ElementV = half;
using LayoutV = layout::RowMajor;
using ElementP = half;
using LayoutP = layout::RowMajor;
using ElementO = half;
using LayoutO = layout::RowMajor;
#endif
#endif
using ElementS = float;
using LayoutS = layout::RowMajor;
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
using BlockMmadQK = TriAttBatchMmad::BlockMmadQK;

using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::NONE, float>;
using PType = Gemm::GemmType<ElementP, LayoutP>;
using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

using L1TileShapePV = GemmShape<128, 128, 256>;
using L0TileShapePV = GemmShape<128, 128, 128>;
using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<false, false>;
using VType = Gemm::GemmType<ElementV, LayoutV>;
using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
using BlockMmadPV = TriAttBatchMmad::BlockMmadPV;

using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, float>;
using OType = Gemm::GemmType<ElementO, LayoutO>;
using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

using DispatchPolicyInitOut = Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
using EpilogueInitOut = Epilogue::Block::BlockEpilogue<DispatchPolicyInitOut, OType, LseType>;
using Kernel = SplitFuse::FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>;
}

extern "C" __global__ __aicore__ void tri_att_fa_official_bias_probe(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR bias1, GM_ADDR bias2,
    GM_ADDR actual_q, GM_ADDR actual_kv, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(FAInferTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KernelCommon::FAIKernelParams params{
        query, key, value, nullptr, nullptr, actual_q, actual_kv,
        output, nullptr, user, tiling, bias1, bias2};
    SplitFuseProbe::Kernel kernel;
    kernel(params);
}

#ifndef ASCENDC_CPU_DEBUG
void tri_att_fa_official_bias_probe_do(uint32_t blockDim, void *l2ctrl, void *stream,
                                       uint8_t *query, uint8_t *key, uint8_t *value,
                                       uint8_t *bias1, uint8_t *bias2,
                                       uint8_t *actual_q, uint8_t *actual_kv, uint8_t *output,
                                       uint8_t *workspace, uint8_t *tiling)
{
    tri_att_fa_official_bias_probe<<<blockDim, l2ctrl, stream>>>(
        query, key, value, bias1, bias2, actual_q, actual_kv, output, workspace, tiling);
}
#endif
