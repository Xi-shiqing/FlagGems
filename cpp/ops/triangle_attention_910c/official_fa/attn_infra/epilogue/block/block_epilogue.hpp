/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "../../../attn_infra/base_defs.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace NpuArch::Epilogue::Block

#include "../../../attn_infra/epilogue/block/block_epilogue_online_softmax.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_online_softmax_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_rescale_o.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_rescale_o_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_init_outputs.hpp"
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP