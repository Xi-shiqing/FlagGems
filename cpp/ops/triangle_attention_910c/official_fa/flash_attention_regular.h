/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_regular.h
 * \brief
 */
#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include "kernel_common.hpp"

using namespace NpuArch;
using namespace KernelCommon;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKernel::MaskType MASK_TYPE = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout INPUT_LAYOUT = FaiKernel::inputLayout::BSND>
    class FAInferKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;

        // Methods
        __aicore__ inline
        FAInferKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
            uint64_t mm2OutSize = fATilingData->mm2OutSize;
            uint32_t batch = fATilingData->batch;
            uint32_t qHeads = fATilingData->numHeads;
            uint32_t kvHeads = fATilingData->kvHeads;
            uint32_t embed = fATilingData->embeddingSize;
            uint32_t embedV = fATilingData->embeddingSizeV;
            uint32_t pagedBlockSize = fATilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = fATilingData->totalTaskNum;
            uint32_t blockSize = fATilingData->blockSize;
            uint32_t maskType = fATilingData->maskType;
            float scaleValue = fATilingData->scaleValue;
            uint32_t pseStride = fATilingData->pseStride;
            uint32_t bias1Stride = fATilingData->bias1Stride;
            uint32_t bias1Rows = fATilingData->bias1Rows;
            uint32_t bias2BatchShared = fATilingData->bias2BatchShared;
            uint32_t bias2GroupsPerBatch = fATilingData->bias2GroupsPerBatch;
            if (bias2GroupsPerBatch == 0U) {
                bias2GroupsPerBatch = 1U;
            }

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            __gm__ uint8_t* currentKey;
            __gm__ uint8_t* currentValue;
            if constexpr (PAGED_CACHE_FLAG) {
                AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)params.k);
                AscendC::ListTensorDesc valueListTensorDescInit((__gm__ void*)params.v);
                currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
                currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            } else {
                currentKey = (__gm__ uint8_t*)params.k;
                currentValue = (__gm__ uint8_t*)params.v;
            }
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementS> gS;
            uint64_t mm1StageOffset = mm1OutSize + smOnlineOutSize + mm2OutSize + mm2OutSize;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace + mm1StageOffset));
            AscendC::GlobalTensor<float> gPse;
            if (params.pse != nullptr) {
                gPse.SetGlobalBuffer((__gm__ float *)params.pse);
            }
            AscendC::GlobalTensor<float> gBias1;
            AscendC::GlobalTensor<float> gBias2;
            if (params.bias1 != nullptr && params.bias2 != nullptr) {
                gBias1.SetGlobalBuffer((__gm__ float *)params.bias1);
                gBias2.SetGlobalBuffer((__gm__ float *)params.bias2);
            }
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef TRIATT_BMM
            AscendC::TPipe tPipe;
            BlockMmadQK blockMmadQK(resource, 0, 0);
            BlockMmadPV blockMmadPV(resource, 0, 0, 0);
            AscendC::tiling::TCubeTiling mm1FullFull;
            AscendC::tiling::TCubeTiling mm2FullFull;
            AscendC::tiling::TCubeTiling mm2FullTail;
            const __gm__ int32_t *mm1Words =
                reinterpret_cast<const __gm__ int32_t *>(&fATilingData->mm1FullFull);
            const __gm__ int32_t *mm2Words =
                reinterpret_cast<const __gm__ int32_t *>(&fATilingData->mm2FullFull);
            const __gm__ int32_t *mm2TailWords =
                reinterpret_cast<const __gm__ int32_t *>(&fATilingData->mm2FullTail);
            int32_t *mm1LocalWords = reinterpret_cast<int32_t *>(&mm1FullFull);
            int32_t *mm2LocalWords = reinterpret_cast<int32_t *>(&mm2FullFull);
            int32_t *mm2TailLocalWords = reinterpret_cast<int32_t *>(&mm2FullTail);
            for (uint32_t i = 0; i < sizeof(mm1FullFull) / sizeof(int32_t); ++i) {
                mm1LocalWords[i] = mm1Words[i];
                mm2LocalWords[i] = mm2Words[i];
                mm2TailLocalWords[i] = mm2TailWords[i];
            }
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(),
                              blockMmadQK.MatmulObject(), &mm1FullFull,
                              blockMmadPV.MatmulObject(), &mm2FullFull,
                              blockMmadPV.MatmulObjectTail(), &mm2FullTail);
#endif
#ifdef __DAV_C220_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            uint32_t kDynNum = NpuArch::Detail::Alignment::RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ?
                NpuArch::Detail::Alignment::RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
#ifndef TRIATT_BMM
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, L1_QK_SIZE);
#endif
 #ifdef TRIATT_BMM
            blockMmadQK.SetTiling(
                &fATilingData->mm1FullFull, &fATilingData->mm1FullTail,
                &fATilingData->mm1TailFull, &fATilingData->mm1TailTail,
                fATilingData->qTileSize, fATilingData->kvStackSize);
            blockMmadPV.SetTiling(
                &fATilingData->mm2FullFull, &fATilingData->mm2FullTail,
                &fATilingData->mm2TailFull, &fATilingData->mm2TailTail,
                fATilingData->qTileSize, fATilingData->kvStackSize);
 #endif
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            EpilogueRescaleO epilogueRescaleO(resource);
            EpilogueInitOut epilogueInitOut(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
#ifdef TRIATT_BNSD
            // This candidate receives the same ABI shape as TND, but the backing
            // storage is BNSD so each head's token rows are contiguous.
            uint64_t strideQ = static_cast<uint64_t>(embed);
            uint64_t strideO = static_cast<uint64_t>(embedV);
            uint64_t strideK = static_cast<uint64_t>(embed);
            uint64_t strideV = static_cast<uint64_t>(embedV);
#else
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
#endif
            uint32_t embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            uint32_t preTotalTaskNum = 0;
            uint32_t curBatch = 0;
            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
            if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                uint32_t prevQSeqlenSum = (curBatch == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            #ifdef TRIATT_BMM
            uint32_t curQNBlockTile = 1U;
#else
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
#endif
            uint32_t qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
#ifdef TRIATT_BMM
            uint32_t curQSBlockTile = fATilingData->qTileSize;
#else
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
#endif
            uint32_t curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curTotalTaskNum = firstBatchTaskNum;

            // Go through each task.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                // Get the offset of each core on the GM.
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
#ifdef TRIATT_BNSD
                    qBOffset += static_cast<uint64_t>(qSeqlen) * qHeads * strideQ;
#else
                    qBOffset += qSeqlen * strideQ;
#endif
                    if constexpr (!PAGED_CACHE_FLAG) {
#ifdef TRIATT_BNSD
#ifdef TRIATT_BMM
                        kBOffset += static_cast<uint64_t>(kvSeqlen) * qHeads * strideK;
                        vBOffset += static_cast<uint64_t>(kvSeqlen) * qHeads * strideV;
#else
                        kBOffset += static_cast<uint64_t>(kvSeqlen) * kvHeads * strideK;
                        vBOffset += static_cast<uint64_t>(kvSeqlen) * kvHeads * strideV;
#endif
                    #else
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
#endif
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
#ifdef TRIATT_BNSD
                    oBOffset += static_cast<uint64_t>(qSeqlen) * qHeads * strideO;
#else
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
#endif
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
#ifdef TRIATT_BMM
                    curQNBlockTile = 1U;
#else
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
#endif
                    qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
#ifdef TRIATT_BMM
                    curQSBlockTile = fATilingData->qTileSize;
#else
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
#endif
                    curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                }
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;
                uint64_t pseOffset = 0;
                if (params.pse != nullptr) {
                    uint64_t pseMatrixSize = static_cast<uint64_t>(pseStride) * pseStride;
                    pseOffset = (static_cast<uint64_t>(curBatch) * qHeads + qNStartIdx) * pseMatrixSize +
                        static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * pseStride;
                }
                uint64_t bias1Offset = 0;
                uint64_t bias2Offset = 0;
                if (params.bias1 != nullptr && params.bias2 != nullptr) {
                    bias1Offset = static_cast<uint64_t>(curBatch) * bias1Stride;
                    uint64_t bias2MatrixSize = static_cast<uint64_t>(qSeqlen) * pseStride;
                    uint64_t bias2Batch = bias2BatchShared ? 0 :
                        (static_cast<uint64_t>(curBatch) / bias2GroupsPerBatch);
                    bias2Offset = (bias2Batch * qHeads + qNStartIdx) * bias2MatrixSize +
                        static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * pseStride;
                }

#ifdef TRIATT_BNSD
                uint64_t gmOffsetQ = qBOffset +
                    static_cast<uint64_t>(qNStartIdx * qSeqlen) * strideQ +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ;
                uint64_t gmOffsetK = kBOffset +
                    static_cast<uint64_t>(kvNIdx * kvSeqlen) * strideK;
                uint64_t gmOffsetV = vBOffset +
                    static_cast<uint64_t>(kvNIdx * kvSeqlen) * strideV;
                uint64_t gmOffsetO = oBOffset +
                    static_cast<uint64_t>(qNStartIdx * qSeqlen) * strideO +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO;
#else
                uint64_t gmOffsetQ = qBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                    static_cast<uint64_t>(qNStartIdx * embed);
                uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
                uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
                uint64_t gmOffsetO = oBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                    static_cast<uint64_t>(qNStartIdx * embedV);
#endif
                uint64_t gmOffsetLse = lseBOffset +
                    static_cast<uint64_t>(lseTokenOffset + qNStartIdx);

                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
                uint32_t rowNum = qSBlockSize * qNBlockSize;
                uint32_t rowNumRound = NpuArch::Detail::Alignment::RoundUp(rowNum, FaiKernel::BLOCK_SIZE);

                int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
                if (maskType != 0U) {
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                }
                uint32_t kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, pagedBlockSize);

                uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
                uint32_t stackSeqTile;
                uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
                uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
                int32_t stackSeqCount = 0;
#ifdef __DAV_C220_VEC__
                if (kvSLoopNumTotal <= 0) {
#ifdef TRIATT_BNSD
                    LayoutO layoutO(qSeqlen, embedV);
#else
                    LayoutO layoutO(qSeqlen, embed * qHeads);
#endif
                    LayoutLse layoutLse(totalQTokens, qHeads);
                    epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutO, layoutLse, qSBlockSize, qNBlockSize);
                }
#endif
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, blockStackNum * pagedBlockSize);
                LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideV);
#ifdef TRIATT_BNSD
                uint32_t qHeadsStride = 1;
 #ifdef TRIATT_BMM
                blockMmadQK.SetRuntimeStrides(fATilingData->qHeadStride, fATilingData->kvHeadStride, qSBlockSize * stackSeqTilePad);
 #endif
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeadsStride);
#else
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
#endif
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                    if (kvSIdx < kvSLoopNumTotal) {
                        if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetS =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK + static_cast<uint64_t>(kvSIdx * pagedBlockSize) * strideK],
                                gS[gmOffsetS],
                                gBlockTable[blockBOffset],
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        } else {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK + static_cast<uint64_t>(kvSIdx * pagedBlockSize) * strideK],
                                gS[gmOffsetS],
                                gBlockTable,
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef TRIATT_BMM
#ifdef __DAV_C220_VEC__
                        LayoutQ layoutQClient(rowNum, embed);
                        LayoutK layoutKClient(strideK, blockStackNum * pagedBlockSize);
                        blockMmadQK.SetRuntimeStrides(
                            fATilingData->qHeadStride, fATilingData->kvHeadStride,
                            qSBlockSize * stackSeqTilePad);
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gK[gmOffsetK + static_cast<uint64_t>(kvSIdx * pagedBlockSize) * strideK],
                            gS[gmOffsetS],
                            gBlockTable,
                            layoutQClient,
                            layoutKClient,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK);
#endif
#endif
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTile);
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                                uint64_t gmOffsetP = gmOffsetS;
                        // causal mask的左上起点
                        uint32_t triUp = noSkipKvS - qSBlockSize;
                        // causal mask的右下止点
                        uint32_t triDown = noSkipKvS;
                        uint32_t kvSStartIdx = kvSIdx * pagedBlockSize;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        // 在causal mask场景下，由mask的左上起点判断当前基块是否需要加mask
                        // 如果实际加mask长度只有1，那么相当于不加mask（主对角线需要被计算）
                        bool doTriUMask = triUp < kvSEndIdx - 1;
                        if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_CAUSAL) {
                            if (doTriUMask) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx);
                            } else {
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                #ifndef TRIATT_BMM
                                Arch::CrossCoreWaitFlag(qkReady);
#endif
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == noMaskStackSeqNum - 1),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod);
                            }
                        } else {
                            #ifndef TRIATT_BMM
                                Arch::CrossCoreWaitFlag(qkReady);
#endif
                            if (params.pse != nullptr) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    0,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    gPse,
                                    pseOffset + static_cast<uint64_t>(kvSStartIdx),
                                    pseStride);
                            } else if (params.bias1 != nullptr && params.bias2 != nullptr) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    0,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    gBias1,
                                    bias1Offset + static_cast<uint64_t>(kvSStartIdx),
                                    bias1Stride,
                                    bias1Rows,
                                    gBias2,
                                    bias2Offset + static_cast<uint64_t>(kvSStartIdx),
                                    pseStride);
                            } else {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    0,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod);
                            }
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCacheCleanAndInvalid<ElementP, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gP);
#endif
                    }
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetOTmp =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;;
 #ifdef TRIATT_BMM
                        blockMmadPV.SetRuntimeStrides(qSBlockSize * stackSeqTilePad, fATilingData->kvHeadStride, qSBlockSize * embedRoundV);
 #endif
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV + static_cast<uint64_t>(nowkvSIdx * pagedBlockSize) * strideV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable[blockBOffset],
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        } else {
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV + static_cast<uint64_t>(nowkvSIdx * pagedBlockSize) * strideV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable,
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef TRIATT_BMM
#ifdef __DAV_C220_VEC__
                        const uint32_t pvHeadCount = qNBlockSize;
                        const uint64_t pvBaseP = static_cast<uint64_t>(
                            coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        const uint64_t pvBaseO = pvBaseP;
                        for (uint32_t pvHeadIdx = 0; pvHeadIdx < pvHeadCount; ++pvHeadIdx) {
                            const uint64_t pvHeadOffsetP = pvBaseP +
                                static_cast<uint64_t>(pvHeadIdx) * qSBlockSize * stackSeqTilePad;
                            const uint64_t pvHeadOffsetO = pvBaseO +
                                static_cast<uint64_t>(pvHeadIdx) * qSBlockSize * embedRoundV;
                            LayoutP layoutPClient(qSBlockSize, stackSeqTile, stackSeqTilePad);
                            LayoutV layoutVClient(stackSeqTilePad, strideV);
                            LayoutOTmp layoutOTmpClient(qSBlockSize, embedV, embedRoundV);
                            blockMmadPV.SetRuntimeStrides(
                                qSBlockSize * stackSeqTilePad, fATilingData->kvHeadStride,
                                qSBlockSize * embedRoundV);
                            GemmCoord actualBlockShapePVHead{qSBlockSize, embedV, stackSeqTile};
                            blockMmadPV(
                                gP[pvHeadOffsetP],
                                gV[gmOffsetV + static_cast<uint64_t>(nowkvSIdx * pagedBlockSize) * strideV],
                                gOTmp[pvHeadOffsetO],
                                gBlockTable,
                                layoutPClient,
                                layoutVClient,
                                layoutOTmpClient,
                                actualBlockShapePVHead,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        }
#endif
#endif
#ifdef __DAV_C220_VEC__
#ifdef TRIATT_BNSD
                        LayoutO layoutO(qSeqlen, embedV);
#else
                        LayoutO layoutO(qSeqlen, embed * qHeads);
#endif
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        #ifndef TRIATT_BMM
                        Arch::CrossCoreWaitFlag(pvReady);
#endif
                        // rescale O
                        epilogueRescaleO(
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                            curStackTileMod);
#endif
                    }
                    stackSeqCount++;
                }
            }
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};
    };
}
#endif
