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
 * \file fused_infer_attention_score_template_tiling_key.h
 * \brief
 */
 
#ifndef FUSED_INFER_ATTENTION_TEMPLATE_TILING_KEY_H_
#define FUSED_INFER_ATTENTION_TEMPLATE_TILING_KEY_H_
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

ASCENDC_TPL_ARGS_DECL(FusedInferAttentionScore, 
    //    bit 8-1 InOutLayoutType(InputLayoutType-OutputLayoutType)
    //    0: InOutLayoutType_BNSD_BNSD
    //    1: InOutLayoutType_BSH_BSH
    //    2: InOutLayoutType_TND_TND
    ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_TND_TND, InOutLayoutType_BSH_BSH),
    //    bit 18-9 Config(S1,S2,D,DV)
    //    0: S1Aligned64_S2Aligned64_DAligned256_DVAligned256,
    //    1: S1Aligned64_S2Aligned64_DAligned512_DVAligned512,    
    //    2: S1Aligned64_S2Aligned256_DAligned64_DVAligned64 ,
    //    3: S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
    //    4: S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
    //    5: S1Aligned128_S2Aligned128_DAligned128_DVAligned64,
    //    6: S1Aligned128_S2Aligned128_DAligned128_DVAligned128,
    //    7: S1Aligned128_S2Aligned128_DAligned192_DVAligned128,
    //    8: S1Aligned128_S2Aligned128_DAligned256_DVAligned128,
    //    9: S1Aligned128_S2Aligned128_DAligned256_DVAligned256,
    //   10: S1Aligned128_S2Aligned128_DAligned512_DVAligned512,
    //   11: S1Aligned128_S2Aligned256_DAligned64_DVAligned64,  
    //   12: Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64,
    //   13: Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128,
    //   14: Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256,
    //   15: Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512
    ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
    //    bit 22-19 PseMode
    //    0: PSE_MODE_PSE_OUTER_MUL_ADD_TYPE
    //    1: PSE_MODE_PSE_OUTER_ADD_MUL_TYPE
    //    2: PSE_MODE_PSE_INNER_MUL_ADD_TYPE
    //    3: PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE
    //    4: PSE_MODE_PSE_INVALID_TYPE
    //    9: PSE_MODE_PSE_NONE_TYPE
    ASCENDC_TPL_UINT_DECL(PseMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_OUTER_ADD_MUL_TYPE,
                        PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_INVALID_TYPE, PSE_MODE_PSE_NONE_TYPE),
    //    bit 27-23 QuantMode
    //    0: PER_CHANNEL
    //    1: PER_TOKEN
    //    2: K_PER_CHANNEL_V_PER_TOKEN
    //    4: PER_TOKEN_PAGE_ATTENTION
    //    5: PER_TOKEN_HEAD_PAGE_ATTENTION
    ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
    //    bit 28 HasAttenMask
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),
    //    bit 29 HasRope
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasRope, false, true),
    //    bit 30 IsPa
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(IsPa, false, true),
    //    bit 31 IsFd
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(IsFd, false, true),
    //    bit 32 EmptyTensor
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EmptyTensor, false, true),
    //    bit 34-33 PFAMask
    //    0: DISABLE_MASK
    //    1: ENABLE_MASK_NO_BAND
    //    2: ENABLE_MASK_BAND
    ASCENDC_TPL_UINT_DECL(PFAMask, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
    //    bit 37-35 PFAMatMulType
    //    0: PFAMatMulType_MM_PFA
    //    1: PFAMatMulType_MM_PA
    //    2: PFAMatMulType_MM_IFA_MLA
    //    3: PFAMatMulType_MM_IFA_MLA_PA
    //    4: PFAMatMulType_MM_PA_D512
    //    5: PFAMatMulType_MM_DN
    ASCENDC_TPL_UINT_DECL(PFAMatMulType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA, PFAMatMulType_MM_PA, PFAMatMulType_MM_IFA_MLA, PFAMatMulType_MM_IFA_MLA_PA,
                        PFAMatMulType_MM_PA_D512, PFAMatMulType_MM_DN),
);

ASCENDC_TPL_SEL(
    // ifa
#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT4 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT8_E5M2 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT4_E2M1 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT4_E1M2 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN,
                            AntiquantMode_PER_TOKEN_PAGE_ATTENTION, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT4 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL, AntiquantMode_PER_TOKEN, AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT8_E5M2 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16) 
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT4_E2M1 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(

        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT4_E1M2 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_CHANNEL),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    ASCENDC_TPL_ARGS_SEL(        
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, false),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA),        
        ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataV2)
    ),
#endif
// pfa
#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E5M2)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E5M2)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_BOOL_SEL(IsPa, false, true),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_UINT_SEL(PFAMask,  ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_INT8 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(

        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                                    Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),  //注意 这里的逻辑有区别，false true对应disable和enable 
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA,  PFAMatMulType_MM_PA),

        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(

        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType,  ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PA_D512, PFAMatMulType_MM_IFA_MLA),

        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif
//全量化使用老模板参数
#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E5M2 && ORIG_DTYPE_KEY == DT_FLOAT8_E5M2 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                                    Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA,  PFAMatMulType_MM_PA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType,  ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PA_D512, PFAMatMulType_MM_IFA_MLA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
     // fp8 pre-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, PerBlock), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif
//全量化使用老模板参数
#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                                    Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),   
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA,  PFAMatMulType_MM_PA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),   
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType,  ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PA_D512, PFAMatMulType_MM_IFA_MLA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
     // fp8 pre-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, PerBlock), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E5M2 && ORIG_DTYPE_KEY == DT_FLOAT8_E5M2 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    // fp8 pre-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, PerBlock), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    // fp8 pre-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, PerBlock), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif
//全量化使用老模板参数
#if (ORIG_DTYPE_QUERY == DT_HIFLOAT8 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                                    Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PFA,  PFAMatMulType_MM_PA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode,ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE),   
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FullQuantMode), //未使用
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(IsPa, 0),  //未使用
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), //未使用
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), //未使用
        ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST,PFAMask_DISABLE_MASK, PFAMask_ENABLE_MASK_NO_BAND, PFAMask_ENABLE_MASK_BAND),
        ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, PFAMatMulType_MM_PA_D512, PFAMatMulType_MM_IFA_MLA),
        ASCENDC_TPL_TILING_STRUCT_SEL(PFAFullQuantTilingData)
    ),
#endif
// 空tensor，必须要有一个使得列表不为空，否则会报无法推导模板参数的错误
ASCENDC_TPL_ARGS_SEL( 
    ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
    ASCENDC_TPL_BOOL_SEL(HasRope, 0),
    ASCENDC_TPL_BOOL_SEL(IsPa, 0),
    ASCENDC_TPL_BOOL_SEL(IsFd, 0),
    ASCENDC_TPL_BOOL_SEL(EmptyTensor, 1),
    ASCENDC_TPL_UINT_SEL(PFAMask, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(PFAMatMulType, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_TILING_STRUCT_SEL(PromptFlashAttentionTilingData)
    ),
);
#endif