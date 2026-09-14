#include "kernel_operator.h"

namespace AscendC {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *) {}
}

namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *) {}
}

using namespace AscendC;

constexpr IsResetLoad3dConfig TRIATT_LOAD3D_CONFIG_FUSED = {true, true};

extern "C" __global__ __aicore__ void tri_att_qk_bias_fused(
    GM_ADDR query, GM_ADDR key, GM_ADDR bias, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    const uint32_t blockId = GetBlockIdx();
    const uint32_t m = tiling_data.m;
    const uint32_t n = tiling_data.n;
    const uint32_t k = tiling_data.k;
    constexpr uint32_t tileM = 128;
    constexpr uint32_t tileN = 64;
    const uint32_t tilesN = (n + tileN - 1) / tileN;
    const uint32_t tileMId = blockId / tilesN;
    const uint32_t tileNId = blockId % tilesN;
    const uint32_t mOffset = tileMId * tileM;
    const uint32_t nOffset = tileNId * tileN;

    if (mOffset + tileM > m || nOffset + tileN > n) {
        return;
    }

#ifdef __DAV_C220_CUBE__
    if ASCEND_IS_AIC {
        GlobalTensor<half> queryGm;
        GlobalTensor<half> keyGm;
        GlobalTensor<float> scoreTmpGm;
        queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query) + mOffset * k);
        keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key) + nOffset * k);
        scoreTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + mOffset * n + nOffset);
        SetMMLayoutTransform(true);

        TPipe pipe;
        TBuf<TPosition::A1> queryL1Buf;
        TBuf<TPosition::B1> keyL1Buf;
        TBuf<TPosition::A2> queryL0Buf;
        TBuf<TPosition::B2> keyL0Buf;
        TBuf<TPosition::CO1> scoreL0Buf;
        pipe.InitBuffer(queryL1Buf, tileM * k * sizeof(half));
        pipe.InitBuffer(keyL1Buf, tileN * k * sizeof(half));
        pipe.InitBuffer(queryL0Buf, tileM * k * sizeof(half));
        pipe.InitBuffer(keyL0Buf, tileN * k * sizeof(half));
        pipe.InitBuffer(scoreL0Buf, tileM * tileN * sizeof(float));

        auto queryL1 = queryL1Buf.Get<half>();
        auto keyL1 = keyL1Buf.Get<half>();
        auto queryL0 = queryL0Buf.Get<half>();
        auto keyL0 = keyL0Buf.Get<half>();
        auto scoreL0 = scoreL0Buf.Get<float>();

        Nd2NzParams queryNd2Nz;
        queryNd2Nz.ndNum = 1;
        queryNd2Nz.nValue = tileM;
        queryNd2Nz.dValue = k;
        queryNd2Nz.srcDValue = k;
        queryNd2Nz.dstNzC0Stride = tileM;
        queryNd2Nz.dstNzNStride = 1;
        queryNd2Nz.srcNdMatrixStride = 0;
        queryNd2Nz.dstNzMatrixStride = 0;
        DataCopy(queryL1, queryGm, queryNd2Nz);

        Nd2NzParams keyNd2Nz;
        keyNd2Nz.ndNum = 1;
        keyNd2Nz.nValue = tileN;
        keyNd2Nz.dValue = k;
        keyNd2Nz.srcDValue = k;
        keyNd2Nz.dstNzC0Stride = tileN;
        keyNd2Nz.dstNzNStride = 1;
        keyNd2Nz.srcNdMatrixStride = 0;
        keyNd2Nz.dstNzMatrixStride = 0;
        DataCopy(keyL1, keyGm, keyNd2Nz);

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        LoadData3DParamsV2<half> queryLoad;
        queryLoad.l1H = 8;
        queryLoad.l1W = 16;
        queryLoad.channelSize = 32;
        queryLoad.padList[0] = 0;
        queryLoad.padList[1] = 0;
        queryLoad.padList[2] = 0;
        queryLoad.padList[3] = 255;
        queryLoad.mExtension = tileM;
        queryLoad.kExtension = k;
        queryLoad.mStartPt = 0;
        queryLoad.kStartPt = 0;
        queryLoad.strideW = 1;
        queryLoad.strideH = 1;
        queryLoad.filterW = 1;
        queryLoad.filterSizeW = (1 >> 8) & 255;
        queryLoad.filterH = 1;
        queryLoad.filterSizeH = (1 >> 8) & 255;
        queryLoad.dilationFilterW = 1;
        queryLoad.dilationFilterH = 1;
        queryLoad.enTranspose = false;
        queryLoad.fMatrixCtrl = false;
        LoadData<half, TRIATT_LOAD3D_CONFIG_FUSED>(queryL0, queryL1, queryLoad);

        LoadData2DParams keyLoad;
        keyLoad.startIndex = 0;
        keyLoad.repeatTimes = k / 4;
        keyLoad.srcStride = 1;
        keyLoad.dstGap = 0;
        keyLoad.ifTranspose = false;
        LoadData(keyL0, keyL1, keyLoad);

        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        MmadParams params;
        params.m = tileM;
        params.n = tileN;
        params.k = k;
        params.cmatrixInitVal = true;
        params.cmatrixSource = false;
        params.unitFlag = 0b11;
        Mmad(scoreL0, queryL0, keyL0, params);
        PipeBarrier<PIPE_M>();

        FixpipeParamsV220 fixParams;
        fixParams.mSize = tileM;
        fixParams.nSize = tileN;
        fixParams.dstStride = n;
        fixParams.srcStride = tileM;
        fixParams.quantPre = QuantMode_t::NoQuant;
        fixParams.reluEn = false;
        fixParams.unitFlag = 3;
        Fixpipe<float, float, CFG_ROW_MAJOR>(scoreTmpGm, scoreL0, fixParams);

        // The AIV side must not read the Fixpipe destination until the write is visible.
        CrossCoreSetFlag<0x2, PIPE_FIX>(1);
    }
#endif

#ifdef __DAV_C220_VEC__
    if ASCEND_IS_AIV {
        GlobalTensor<float> scoreGm;
        GlobalTensor<float> biasGm;
        GlobalTensor<float> outGm;
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + mOffset * n + nOffset);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias) + mOffset * n + nOffset);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + mOffset * n + nOffset);

        CrossCoreWaitFlag(1);

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

        SetVectorMask<int8_t>((tileM * tileN * sizeof(float)) / sizeof(int8_t));
        Add<float, false>(outUb, scoreUb, biasUb, static_cast<uint64_t>(0), tileM * tileN / 64,
                          BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        const DataCopyParams outParams(tileM, tileN * sizeof(float) / 32, 0, rowGapBlocks);
        DataCopy(outGm, outUb, outParams);
        PipeBarrier<PIPE_ALL>();
    }
#endif
}

#ifndef ASCENDC_CPU_DEBUG
void tri_att_qk_bias_fused_do(uint32_t blockDim, void *l2ctrl, void *stream,
                              uint8_t *query, uint8_t *key, uint8_t *bias, uint8_t *out,
                              uint8_t *workspace, uint8_t *tiling)
{
    tri_att_qk_bias_fused<<<blockDim, l2ctrl, stream>>>(query, key, bias, out, workspace, tiling);
}
#endif
