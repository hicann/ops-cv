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
 * \file crop_and_resize_apt.cpp
 * \brief Kernel entry for crop_and_resize operator
 *
 * Single template parameter:
 *   schMode (uint32_t): scene mode (0 = bilinear)
 * dtype is auto-instantiated via DTYPE_ macros (DTYPE_X, DTYPE_BOXES, DTYPE_Y).
 */

#include "kernel_tiling/kernel_tiling.h"
#include "arch35/crop_and_resize_tiling_key.h"
#include "arch35/crop_and_resize_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void crop_and_resize(GM_ADDR x, GM_ADDR boxes, GM_ADDR box_index, GM_ADDR crop_size, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CropAndResizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(CropAndResizeTilingData, tilingData, tiling);

    if constexpr (schMode == CROP_AND_RESIZE_MODE_BILINEAR) {
        // DTYPE_ 宏自动实例化所有 dtype 组合
        // DTYPE_X -> x 的 dtype (half/float)
        // DTYPE_BOXES -> boxes 的 dtype (half/float)
        // DTYPE_Y -> y 的 dtype (= boxes dtype)
        NsCropAndResize::Process<DTYPE_X, DTYPE_BOXES, DTYPE_Y>(x, boxes, box_index, crop_size, y, &tilingData);
    }
}
