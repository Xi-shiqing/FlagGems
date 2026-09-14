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
 * \file fused_infer_attention_score_v3.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch32/fused_infer_attention_score_tilingkey.h"

#ifdef NOT_DYNAMIC_COMPILE
#include "../../common/op_kernel/arch32/fia_kernel_empty_tensor.h"
#else
#include "../common/arch32/fia_kernel_empty_tensor.h"
#endif

#ifdef FIA_ENABLE_MLA
// mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'FusedInferAttentionScoreTilingData'
#if ((ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) &&                                   \
     (ORIG_DTYPE_KEY == DT_FLOAT16)) ||                                                                                \
    ((ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16))
#ifdef NOT_DYNAMIC_COMPILE
#include "../../common/op_kernel/arch32/fia_kernel_nonquant_mla.h"
#include "../../common/op_kernel/arch32/fia_kernel_nonquant.h"
#else
#include "../common/arch32/fia_kernel_nonquant_mla.h"
#include "../common/arch32/fia_kernel_nonquant.h"
#endif
#endif
#endif // FIA_ENABLE_MLA

using namespace AscendC;

#define INVOKE_FIA_OP_GENERAL_IMPL(templateClass, CubeBlockType, VecBlockType, FdBlockType, ...)                       \
    do {                                                                                                               \
        using CubeBlockTypeT = CubeBlockType<FIAType<__VA_ARGS__>>;                                                    \
        using VecBlockTypeT = VecBlockType<FIAType<__VA_ARGS__>>;                                                      \
        using FdBlockTypeT = FdBlockType<FIAType<__VA_ARGS__>>;                                                        \
                                                                                                                       \
        templateClass<FIAType<__VA_ARGS__>, CubeBlockTypeT, VecBlockTypeT, FdBlockTypeT> op;                           \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1,   \
                deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize,   \
                kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,       \
                keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,  \
                learnableSink, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define FIA_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                 \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

extern "C" __global__ __aicore__ void fused_infer_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    // fp16 7buf_nz
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // fp16 7buf_nd
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla PA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_TND
    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    
    TILING_KEY_LIST(QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Gqa NoQuant PA
    TILING_KEY_IS(103000000000200000);
    TILING_KEY_IS(103000000000300000);
    TILING_KEY_IS(103000000000600000);
    TILING_KEY_IS(103000000000700000);
    TILING_KEY_IS(103000000010200000);
    TILING_KEY_IS(103000000010300000);
    TILING_KEY_IS(103000000010600000);
    TILING_KEY_IS(103000000010700000);
    TILING_KEY_IS(103000000020200000);
    TILING_KEY_IS(103000000020300000);
    TILING_KEY_IS(103000000020600000);
    TILING_KEY_IS(103000000020700000);

    TILING_KEY_IS(103000000000200001);
    TILING_KEY_IS(103000000000300001);
    TILING_KEY_IS(103000000000600001);
    TILING_KEY_IS(103000000000700001);
    TILING_KEY_IS(103000000010200001);
    TILING_KEY_IS(103000000010300001);
    TILING_KEY_IS(103000000010600001);
    TILING_KEY_IS(103000000010700001);
    TILING_KEY_IS(103000000020200001);
    TILING_KEY_IS(103000000020300001);
    TILING_KEY_IS(103000000020600001);
    TILING_KEY_IS(103000000020700001);

    TILING_KEY_IS(103000000000200003);
    TILING_KEY_IS(103000000000300003);
    TILING_KEY_IS(103000000010200003);
    TILING_KEY_IS(103000000010300003);
    TILING_KEY_IS(103000000020200003);
    TILING_KEY_IS(103000000020300003);

    TILING_KEY_IS(103000000000200005);
    TILING_KEY_IS(103000000000300005);
    TILING_KEY_IS(103000000010200005);
    TILING_KEY_IS(103000000010300005);
    TILING_KEY_IS(103000000020200005);
    TILING_KEY_IS(103000000020300005);

    // Gqa NoQuant Non PA Non Perf
    TILING_KEY_IS(103000000000000000);
    TILING_KEY_IS(103000000010000001);
    TILING_KEY_IS(103000000030000003);
    TILING_KEY_IS(103000000050000005);
    TILING_KEY_IS(103000000000100000);
    TILING_KEY_IS(103000000010100001);
    TILING_KEY_IS(103000000030100003);
    TILING_KEY_IS(103000000050100005);
    TILING_KEY_IS(103000000000400000);
    TILING_KEY_IS(103000000010400001);
    TILING_KEY_IS(103000000000500000);
    TILING_KEY_IS(103000000010500001);

    // Gqa NoQuant PA 泛化
    TILING_KEY_IS(104000000000200000);
    TILING_KEY_IS(104000000000300000);
    TILING_KEY_IS(104000000000600000);
    TILING_KEY_IS(104000000000700000);
    TILING_KEY_IS(104000000010200000);
    TILING_KEY_IS(104000000010300000);
    TILING_KEY_IS(104000000010600000);
    TILING_KEY_IS(104000000010700000);
    TILING_KEY_IS(104000000020200000);
    TILING_KEY_IS(104000000020300000);
    TILING_KEY_IS(104000000020600000);
    TILING_KEY_IS(104000000020700000);

    TILING_KEY_IS(104000000000200001);
    TILING_KEY_IS(104000000000300001);
    TILING_KEY_IS(104000000000600001);
    TILING_KEY_IS(104000000000700001);
    TILING_KEY_IS(104000000010200001);
    TILING_KEY_IS(104000000010300001);
    TILING_KEY_IS(104000000010600001);
    TILING_KEY_IS(104000000010700001);
    TILING_KEY_IS(104000000020200001);
    TILING_KEY_IS(104000000020300001);
    TILING_KEY_IS(104000000020600001);
    TILING_KEY_IS(104000000020700001);

    TILING_KEY_IS(104000000000200003);
    TILING_KEY_IS(104000000000300003);
    TILING_KEY_IS(104000000000600003);
    TILING_KEY_IS(104000000000700003);
    TILING_KEY_IS(104000000010200003);
    TILING_KEY_IS(104000000010300003);
    TILING_KEY_IS(104000000010600003);
    TILING_KEY_IS(104000000010700003);
    TILING_KEY_IS(104000000020200003);
    TILING_KEY_IS(104000000020300003);
    TILING_KEY_IS(104000000020600003);
    TILING_KEY_IS(104000000020700003);

    TILING_KEY_IS(104000000000200005);
    TILING_KEY_IS(104000000000300005);
    TILING_KEY_IS(104000000000600005);
    TILING_KEY_IS(104000000000700005);
    TILING_KEY_IS(104000000010200005);
    TILING_KEY_IS(104000000010300005);
    TILING_KEY_IS(104000000010600005);
    TILING_KEY_IS(104000000010700005);
    TILING_KEY_IS(104000000020200005);
    TILING_KEY_IS(104000000020300005);
    TILING_KEY_IS(104000000020600005);
    TILING_KEY_IS(104000000020700005);

    // Gqa NoQuant Non PA Non Perf 泛化
    TILING_KEY_IS(104000000000000000);
    TILING_KEY_IS(104000000010000001);
    TILING_KEY_IS(104000000030000003);
    TILING_KEY_IS(104000000050000005);
    TILING_KEY_IS(104000000000100000);
    TILING_KEY_IS(104000000010100001);
    TILING_KEY_IS(104000000030100003);
    TILING_KEY_IS(104000000050100005);
    TILING_KEY_IS(104000000000400000);
    TILING_KEY_IS(104000000010400001);
    TILING_KEY_IS(104000000030400003);
    TILING_KEY_IS(104000000050400005);
    TILING_KEY_IS(104000000000500000);
    TILING_KEY_IS(104000000010500001);
    TILING_KEY_IS(104000000030500003);
    TILING_KEY_IS(104000000050500005);
// Mla PA fp16 kv_NZ
#if (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// Mla PA fp16 kv_BNSD
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
// Mla PA fp16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
// Mla NoPA fp16 kv_BNSD
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA fp16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
// Mla NoPA fp16 kv_TND
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
// Gqa NoQuant PA Non Perf
#elif TILING_KEY_VAR == 103000000000200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000000700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000010700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000020200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 103000000020700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 103000000000200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000000700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000010700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000020200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 103000000020700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000000200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000020200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000000200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000000300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000010300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000020200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 103000000020300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ,
                               false);
// Gqa NoQuant Non PA Non Perf
#elif TILING_KEY_VAR == 103000000000000000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010000001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000030000003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false);
#elif TILING_KEY_VAR == 103000000050000005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               false);
#elif TILING_KEY_VAR == 103000000000100000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010100001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000030100003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false);
#elif TILING_KEY_VAR == 103000000050100005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               false);
#elif TILING_KEY_VAR == 103000000000400000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010400001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000000500000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010500001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);

// Gqa NoQuant PA Non Perf 泛化
#elif TILING_KEY_VAR == 104000000000200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000000700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000010700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000020200000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020300000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020600000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 104000000020700000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 104000000000200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000000700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000010700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000020200001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020300001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020600001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 104000000020700001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000000200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000600003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000000700003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010600003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000010700003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000020200003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020300003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020600003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 104000000020700003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000000200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000000600005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000000700005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000010600005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000010700005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000020200005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020300005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ,
                               false);
#elif TILING_KEY_VAR == 104000000020600005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ,
                               true);
#elif TILING_KEY_VAR == 104000000020700005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NZ, true);
// Gqa NoQuant Non PA Non Perf 泛化
#elif TILING_KEY_VAR == 104000000000000000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000010000001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000030000003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false);
#elif TILING_KEY_VAR == 104000000050000005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               false);
#elif TILING_KEY_VAR == 104000000000100000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 104000000010100001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 104000000030100003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false);
#elif TILING_KEY_VAR == 104000000050100005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               false);
#elif TILING_KEY_VAR == 104000000000400000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010400001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000030400003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               true);
#elif TILING_KEY_VAR == 104000000050400005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               true);
#elif TILING_KEY_VAR == 104000000000500000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 104000000010500001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 104000000030500003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               true);
#elif TILING_KEY_VAR == 104000000050500005
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::NTD, false, false, FIA_LAYOUT::NTD,
                               true);

#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)
    // bfl6 7buf_nz
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // 7buf_nd
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla PA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_TND
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Gqa NoQuant PA
    TILING_KEY_IS(103000000000222220);
    TILING_KEY_IS(103000000000322220);
    TILING_KEY_IS(103000000000622220);
    TILING_KEY_IS(103000000000722220);
    TILING_KEY_IS(103000000010222220);
    TILING_KEY_IS(103000000010322220);
    TILING_KEY_IS(103000000010622220);
    TILING_KEY_IS(103000000010722220);
    TILING_KEY_IS(103000000020222220);
    TILING_KEY_IS(103000000020322220);
    TILING_KEY_IS(103000000020622220);
    TILING_KEY_IS(103000000020722220);

    TILING_KEY_IS(103000000000222221);
    TILING_KEY_IS(103000000000322221);
    TILING_KEY_IS(103000000000622221);
    TILING_KEY_IS(103000000000722221);
    TILING_KEY_IS(103000000010222221);
    TILING_KEY_IS(103000000010322221);
    TILING_KEY_IS(103000000010622221);
    TILING_KEY_IS(103000000010722221);
    TILING_KEY_IS(103000000020222221);
    TILING_KEY_IS(103000000020322221);
    TILING_KEY_IS(103000000020622221);
    TILING_KEY_IS(103000000020722221);

    TILING_KEY_IS(103000000000222223);
    TILING_KEY_IS(103000000000322223);
    TILING_KEY_IS(103000000010222223);
    TILING_KEY_IS(103000000010322223);
    TILING_KEY_IS(103000000020222223);
    TILING_KEY_IS(103000000020322223);

    TILING_KEY_IS(103000000000222225);
    TILING_KEY_IS(103000000000322225);
    TILING_KEY_IS(103000000010222225);
    TILING_KEY_IS(103000000010322225);
    TILING_KEY_IS(103000000020222225);
    TILING_KEY_IS(103000000020322225);


    // Gqa NoQuant Non PA
    TILING_KEY_IS(103000000000022220);
    TILING_KEY_IS(103000000010022221);
    TILING_KEY_IS(103000000030022223);
    TILING_KEY_IS(103000000000122220);
    TILING_KEY_IS(103000000010122221);
    TILING_KEY_IS(103000000030122223);
    TILING_KEY_IS(103000000000422220);
    TILING_KEY_IS(103000000010422221);
    TILING_KEY_IS(103000000000522220);
    TILING_KEY_IS(103000000010522221);
    TILING_KEY_IS(103000000050022225);
    TILING_KEY_IS(103000000050122225);

    // Gqa NoQuant PA 泛化
    TILING_KEY_IS(104000000000222220);
    TILING_KEY_IS(104000000000322220);
    TILING_KEY_IS(104000000000622220);
    TILING_KEY_IS(104000000000722220);
    TILING_KEY_IS(104000000010222220);
    TILING_KEY_IS(104000000010322220);
    TILING_KEY_IS(104000000010622220);
    TILING_KEY_IS(104000000010722220);
    TILING_KEY_IS(104000000020222220);
    TILING_KEY_IS(104000000020322220);
    TILING_KEY_IS(104000000020622220);
    TILING_KEY_IS(104000000020722220);

    TILING_KEY_IS(104000000000222221);
    TILING_KEY_IS(104000000000322221);
    TILING_KEY_IS(104000000000622221);
    TILING_KEY_IS(104000000000722221);
    TILING_KEY_IS(104000000010222221);
    TILING_KEY_IS(104000000010322221);
    TILING_KEY_IS(104000000010622221);
    TILING_KEY_IS(104000000010722221);
    TILING_KEY_IS(104000000020222221);
    TILING_KEY_IS(104000000020322221);
    TILING_KEY_IS(104000000020622221);
    TILING_KEY_IS(104000000020722221);

    TILING_KEY_IS(104000000000222223);
    TILING_KEY_IS(104000000000322223);
    TILING_KEY_IS(104000000000622223);
    TILING_KEY_IS(104000000000722223);
    TILING_KEY_IS(104000000010222223);
    TILING_KEY_IS(104000000010322223);
    TILING_KEY_IS(104000000010622223);
    TILING_KEY_IS(104000000010722223);
    TILING_KEY_IS(104000000020222223);
    TILING_KEY_IS(104000000020322223);
    TILING_KEY_IS(104000000020622223);
    TILING_KEY_IS(104000000020722223);

    TILING_KEY_IS(104000000000222225);
    TILING_KEY_IS(104000000000322225);
    TILING_KEY_IS(104000000000622225);
    TILING_KEY_IS(104000000000722225);
    TILING_KEY_IS(104000000010222225);
    TILING_KEY_IS(104000000010322225);
    TILING_KEY_IS(104000000010622225);
    TILING_KEY_IS(104000000010722225);
    TILING_KEY_IS(104000000020222225);
    TILING_KEY_IS(104000000020322225);
    TILING_KEY_IS(104000000020622225);
    TILING_KEY_IS(104000000020722225);


    // Gqa NoQuant Non PA 泛化
    TILING_KEY_IS(104000000000022220);
    TILING_KEY_IS(104000000010022221);
    TILING_KEY_IS(104000000030022223);
    TILING_KEY_IS(104000000050022223);
    TILING_KEY_IS(104000000000122220);
    TILING_KEY_IS(104000000010122221);
    TILING_KEY_IS(104000000030122223);
    TILING_KEY_IS(104000000050122223);
    TILING_KEY_IS(104000000000422220);
    TILING_KEY_IS(104000000010422221);
    TILING_KEY_IS(104000000030422223);
    TILING_KEY_IS(104000000050422223);
    TILING_KEY_IS(104000000000522220);
    TILING_KEY_IS(104000000010522221);
    TILING_KEY_IS(104000000030522223);
    TILING_KEY_IS(104000000050522223);
    TILING_KEY_IS(104000000050022225);
    TILING_KEY_IS(104000000050122225);

    // Mla PA bf16 kv_NZ
#if (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// Mla PA bf16 kv_BNSD
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
// Mla PA bf16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
// Mla NoPA bf16 kv_BNSD
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA bf16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
// Mla NoPA bf16 kv_TND
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);

// Gqa NoQuant PA
#elif TILING_KEY_VAR == 103000000000222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ, true);

#elif TILING_KEY_VAR == 103000000000222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ, true);

#elif TILING_KEY_VAR == 103000000000222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000020222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ);

#elif TILING_KEY_VAR == 103000000000222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000020222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ);

// Gqa NoQuant PA 泛化
#elif TILING_KEY_VAR == 104000000000222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000000722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000010722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000020222220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020322220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020622220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000020722220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::NZ, true);

#elif TILING_KEY_VAR == 104000000000222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000000722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000010722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000020222221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020322221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020622221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000020722221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::NZ, true);

#elif TILING_KEY_VAR == 104000000000222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000622223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000000722223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010622223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000010722223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000020222223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020322223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020622223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000020722223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::NZ, true);

#elif TILING_KEY_VAR == 104000000000222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000000622225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000000722225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000010622225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000010722225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000020222225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020322225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 104000000020622225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 104000000020722225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NZ, true);
// Gqa NoQuant Non PA
#elif TILING_KEY_VAR == 103000000000022220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010022221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000030022223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 103000000050022225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 103000000000122220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010122221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000030122223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 103000000050122225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 103000000000422220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010422221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000050422225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD, true);
#elif TILING_KEY_VAR == 103000000000522220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010522221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000050522225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD, true);
// Gqa NoQuant Non PA 泛化
#elif TILING_KEY_VAR == 104000000000022220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000010022221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000030022223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 104000000050022225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 104000000000122220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 104000000010122221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 104000000030122223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 104000000050122225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 104000000000422220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010422221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000030422223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND, true);
#elif TILING_KEY_VAR == 104000000050422225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD, true);
#elif TILING_KEY_VAR == 104000000000522220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 104000000010522221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 104000000030522223
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND, true);
#elif TILING_KEY_VAR == 104000000050522225
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuant, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false,
                               false, FIA_LAYOUT::NTD, true);
#endif
#endif

// empty tensor 模板
    TILING_KEY_IS(100000000000000020);
#if TILING_KEY_VAR == 100000000000000020
    FiaKernelEmptyTensor<half> op;
    FIA_COPY_TILING_DATA(FusedInferAttentionScoreEmptyTensorTilingData, tiling);
    op.Init(attentionOut, softmaxLse, tiling_data, &tPipe);
    op.Process();
#endif

#endif
}
