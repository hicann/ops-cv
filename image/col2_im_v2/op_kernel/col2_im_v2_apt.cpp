/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file col2_im_v2_apt.cpp
 * \brief col2_im_v2 kernel entry（参数顺序与 REG_OP(Col2ImV2) 原型严格一致：INPUTs + OUTPUTs）
 */

#include "arch35/col2_im_v2_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void col2_im_v2(GM_ADDR x, GM_ADDR output_size, GM_ADDR kernel_size, GM_ADDR y, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Col2ImV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Col2ImV2TilingData, tilingData, tiling);
    // 单一场景模式；output_size/kernel_size 为 const tensor，值已在 tiling 阶段写入 TilingData，GM 地址仅占位
    if constexpr (schMode == COL2_IM_V2_SCH_MODE_DEFAULT) {
        NsCol2ImV2::Process<DTYPE_X>(x, y, &tilingData);
    }
}
