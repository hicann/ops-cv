/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_linear.cpp
 * \brief resize_linear
 */
#include "./arch35/resize_linear_simt_ncl.h"
#include "./arch35/resize_linear_tiling_key.h"

using namespace AscendC;
using namespace ResizeLinear;

template <uint64_t schId, uint64_t isInt32, uint64_t isHalfPixel>
__global__ __aicore__ void resize_linear(GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (isInt32 == 1) {
        ResizeLinear::ResizeLinearSimtNCL<DTYPE_X, uint32_t, isHalfPixel, schId> op;
        op.Init(x, size, y, &tilingData);
        op.Process();
    } else {
        ResizeLinear::ResizeLinearSimtNCL<DTYPE_X, uint64_t, isHalfPixel, schId> op;
        op.Init(x, size, y, &tilingData);
        op.Process();
    }
}
