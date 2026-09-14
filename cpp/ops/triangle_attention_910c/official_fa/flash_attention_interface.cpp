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
 * \file flash_attention_interface.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "flash_attention_regular.h"
using namespace NpuArch;

namespace SplitFuse {
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKernel::MaskType maskCategory = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout inLayout = FaiKernel::inputLayout::TND,
        Epilogue::LseMode lseMode = Epilogue::LseMode::NONE>
    __global__ __aicore__ void FAInfer(
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR mask,
        GM_ADDR blockTables,
        GM_ADDR o,
        GM_ADDR lse,
        GM_ADDR actualQseqlen,
        GM_ADDR actualKvseqlen,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        using ArchTag = Arch::AtlasA2;
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeKv;
        using LayoutK = layout::ColumnMajor;
        using ElementV = InputDtypeKv;
        using LayoutV = layout::RowMajor;
        using ElementS = IntermCalcPrec;
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;
        using LayoutP = layout::RowMajor;
        using ElementO = InputDtypeQ;
        using LayoutO = layout::RowMajor;
        using ElementLse = float;
        using LayoutLse = layout::RowMajor;
        using ElementMask = int8_t;
        using LayoutMask = layout::RowMajor;
        using ElementOTmp = IntermCalcPrec;
        using LayoutOTmp = layout::RowMajor;
        using ElementUpdate = IntermCalcPrec;
        using LayoutUpdate = layout::RowMajor;

        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
        using L0TileShapeQK = GemmShape<128, 128, 128>;
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<PagedCacheFlag, false>;
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;

        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, IntermCalcPrec>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
        using EpilogueOnlineSoftmax =
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

        using L1TileShapePV = GemmShape<128, 128, 256>;
        using L0TileShapePV = GemmShape<128, 128, 128>;
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<PagedCacheFlag, false>;
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;

        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, IntermCalcPrec>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueRescaleO =
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

        using DispatchPolicyInitOutWhenZero = Epilogue::EpilogueAtlasA2InitOutWhenZero<lseMode>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueInitOut =
            Epilogue::Block::BlockEpilogue<DispatchPolicyInitOutWhenZero, OType, LseType>;

        using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV,
                                            EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut,
                                            PagedCacheFlag, maskCategory, inLayout>;
        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
        FAInferKernel flashAttnInfer;
        flashAttnInfer(params);
    }
}