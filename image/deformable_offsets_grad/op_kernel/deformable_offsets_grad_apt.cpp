/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_offsets_grad.cpp
 * \brief deformable_offsets_grad kernel main
 */
#include "arch35/deformable_offsets_grad.h"

#define TILING_KEY_SIMIT_MODE_BFLOAT16 1000
#define TILING_KEY_SIMIT_MODE_FLOAT16 1001
#define TILING_KEY_SIMIT_MODE_FLOAT32 1002

using namespace DeformableOffsetsGrad;
extern "C" __global__ __aicore__ void deformable_offsets_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR offsets, GM_ADDR grad_x, GM_ADDR grad_offsets, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_SIMIT_MODE_FLOAT32)) {
        DeformableOffsetsGrad::DeformableOffsetGrad<float> DeformableOffsetGradObject;
        DeformableOffsetGradObject.Init(grad, x, offsets, grad_x, grad_offsets, &tilingData);
        DeformableOffsetGradObject.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMIT_MODE_FLOAT16)) {
        DeformableOffsetsGrad::DeformableOffsetGrad<half> DeformableOffsetGradObject;
        DeformableOffsetGradObject.Init(grad, x, offsets, grad_x, grad_offsets, &tilingData);
        DeformableOffsetGradObject.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMIT_MODE_BFLOAT16)) {
        DeformableOffsetsGrad::DeformableOffsetGrad<bfloat16_t> DeformableOffsetGradObject;
        DeformableOffsetGradObject.Init(grad, x, offsets, grad_x, grad_offsets, &tilingData);
        DeformableOffsetGradObject.Process();
    }
}