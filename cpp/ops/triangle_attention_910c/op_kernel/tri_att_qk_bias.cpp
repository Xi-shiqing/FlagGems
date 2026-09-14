#include "kernel_operator.h"

namespace AscendC {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *) {}
}

using namespace AscendC;

extern "C" __global__ __aicore__ void tri_att_qk_bias(GM_ADDR score, GM_ADDR bias,
                                                        GM_ADDR out, GM_ADDR workspace,
                                                        GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t blockId = GetBlockIdx();
    const uint32_t m = tiling_data.m;
    const uint32_t n = tiling_data.n;
    const uint32_t tileM = 64;
    const uint32_t tileN = 64;
    const uint32_t tilesN = (n + tileN - 1) / tileN;
    const uint32_t tileMId = blockId / tilesN;
    const uint32_t tileNId = blockId % tilesN;
    const uint32_t mOffset = tileMId * tileM;
    const uint32_t nOffset = tileNId * tileN;

    // The caller pads non-aligned shapes before entering this fixed-tile probe.
    if (mOffset + tileM > m || nOffset + tileN > n) {
        return;
    }

    GlobalTensor<float> scoreGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> outGm;
    scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(score) + mOffset * n + nOffset);
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias) + mOffset * n + nOffset);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + mOffset * n + nOffset);

    TPipe pipe;
    TBuf<TPosition::VECCALC> scoreUbBuf;
    TBuf<TPosition::VECCALC> biasUbBuf;
    TBuf<TPosition::VECCALC> outUbBuf;
    pipe.InitBuffer(scoreUbBuf, tileM * tileN * sizeof(float));
    pipe.InitBuffer(biasUbBuf, tileM * tileN * sizeof(float));
    pipe.InitBuffer(outUbBuf, tileM * tileN * sizeof(float));

    auto scoreUb = scoreUbBuf.Get<float>();
    auto biasUb = biasUbBuf.Get<float>();
    auto outUb = outUbBuf.Get<float>();

    const uint16_t rowGapBlocks = static_cast<uint16_t>((n - tileN) * sizeof(float) / 32);
    const DataCopyParams inParams(tileM, tileN * sizeof(float) / 32, rowGapBlocks, 0);
    DataCopy(scoreUb, scoreGm, inParams);
    DataCopy(biasUb, biasGm, inParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    Add<float, false>(
        outUb,
        scoreUb,
        biasUb,
        (uint64_t)0,
        tileM * tileN / 64,
        BinaryRepeatParams(1, 1, 1, 8, 8, 8)
    );
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    const DataCopyParams outParams(tileM, tileN * sizeof(float) / 32, 0, rowGapBlocks);
    DataCopy(outGm, outUb, outParams);

    (void)workspace;
    (void)tiling;
}

#ifndef ASCENDC_CPU_DEBUG
void tri_att_qk_bias_do(uint32_t blockDim, void *l2ctrl, void *stream,
                        uint8_t *score, uint8_t *bias, uint8_t *out,
                        uint8_t *workspace, uint8_t *tiling)
{
    tri_att_qk_bias<<<blockDim, l2ctrl, stream>>>(score, bias, out, workspace, tiling);
}
#endif
