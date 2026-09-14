#include "kernel_operator.h"

namespace AscendC {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *) {}
}

using namespace AscendC;

constexpr IsResetLoad3dConfig TRIATT_LOAD3D_CONFIG = {true, true};

extern "C" __global__ __aicore__ void tri_att_qk(GM_ADDR query, GM_ADDR key,
                                                  GM_ADDR score, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t blockId = GetBlockIdx();
    const uint32_t m = tiling_data.m;
    const uint32_t n = tiling_data.n;
    const uint32_t k = tiling_data.k;
    const uint32_t tileM = 128;
    const uint32_t tileN = 64;
    const uint32_t tilesN = (n + tileN - 1) / tileN;
    const uint32_t tileMId = blockId / tilesN;
    const uint32_t tileNId = blockId % tilesN;
    const uint32_t mOffset = tileMId * tileM;
    const uint32_t nOffset = tileNId * tileN;
    (void)blockId;
    TPipe pipe;
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyGm;
    GlobalTensor<float> scoreGm;
    queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query) + mOffset * k);
    keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key) + nOffset * k);
    scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(score) + mOffset * n + nOffset);
    SetMMLayoutTransform(true);

    TBuf<TPosition::A1> queryL1Buf;
    TBuf<TPosition::B1> keyL1Buf;
    TBuf<TPosition::A2> queryL0Buf;
    TBuf<TPosition::B2> keyL0Buf;
    TBuf<TPosition::CO1> scoreL0Buf;
    pipe.InitBuffer(queryL1Buf, 128 * 32 * sizeof(half));
    pipe.InitBuffer(keyL1Buf, 64 * 32 * sizeof(half));
    pipe.InitBuffer(queryL0Buf, 128 * 32 * sizeof(half));
    pipe.InitBuffer(keyL0Buf, 64 * 32 * sizeof(half));
    pipe.InitBuffer(scoreL0Buf, 128 * 64 * sizeof(float));

    auto queryL1 = queryL1Buf.Get<half>();
    auto keyL1 = keyL1Buf.Get<half>();
    auto queryL0 = queryL0Buf.Get<half>();
    auto keyL0 = keyL0Buf.Get<half>();
    auto scoreL0 = scoreL0Buf.Get<float>();

    Nd2NzParams queryNd2Nz;
    queryNd2Nz.ndNum = 1;
    queryNd2Nz.nValue = 128;
    queryNd2Nz.dValue = 32;
    queryNd2Nz.srcDValue = 32;
    queryNd2Nz.dstNzC0Stride = 128;
    queryNd2Nz.dstNzNStride = 1;
    queryNd2Nz.srcNdMatrixStride = 0;
    queryNd2Nz.dstNzMatrixStride = 0;
    DataCopy(queryL1, queryGm, queryNd2Nz);

    Nd2NzParams keyNd2Nz;
    keyNd2Nz.ndNum = 1;
    keyNd2Nz.nValue = 64;
    keyNd2Nz.dValue = 32;
    keyNd2Nz.srcDValue = 32;
    keyNd2Nz.dstNzC0Stride = 64;
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
    queryLoad.mExtension = 128;
    queryLoad.kExtension = 32;
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
    LoadData<half, TRIATT_LOAD3D_CONFIG>(queryL0, queryL1, queryLoad);

    LoadData2DParams keyLoad;
    keyLoad.startIndex = 0;
    keyLoad.repeatTimes = 8;
    keyLoad.srcStride = 1;
    keyLoad.dstGap = 0;
    keyLoad.ifTranspose = false;
    LoadData(keyL0, keyL1, keyLoad);

    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

    MmadParams params;
    params.m = 128;
    params.n = 64;
    params.k = 32;
    params.cmatrixInitVal = true;
    params.cmatrixSource = false;
    params.unitFlag = 0b11;
    Mmad(scoreL0, queryL0, keyL0, params);

    PipeBarrier<PIPE_M>();

    DataCopyCO12DstParams outParams;
    outParams.mSize = 128;
    outParams.nSize = 64;
    outParams.dstStride = n;
    outParams.srcStride = 128;
    outParams.quantPre = QuantMode_t::NoQuant;
    outParams.reluPre = 0;
    outParams.channelSplit = false;
    outParams.nz2ndEn = true;
    outParams.unitFlag = 3;
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(scoreGm, scoreL0, outParams);

    (void)workspace;
    (void)tiling;
}

#ifndef ASCENDC_CPU_DEBUG
void tri_att_qk_do(uint32_t blockDim, void *l2ctrl, void *stream,
                   uint8_t *query, uint8_t *key, uint8_t *score,
                   uint8_t *workspace, uint8_t *tiling)
{
    tri_att_qk<<<blockDim, l2ctrl, stream>>>(query, key, score, workspace, tiling);
}
#endif
