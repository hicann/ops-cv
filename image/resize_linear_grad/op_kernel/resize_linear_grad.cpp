/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_linear_grad.cpp
 * \brief resize_linear_grad
 */

#include "./arch35/resize_linear_grad_simt_base.h"
#include "./arch35/resize_linear_grad_tensor_move.h"
#include "./arch35/resize_linear_grad_out1.h"
#include "./arch35/resize_linear_grad_out1_half.h"
#include "./arch35/resize_linear_grad_simt_no_determine.h"
#include "./arch35/resize_linear_grad_simt_determine.h"
#include "./arch35/resize_linear_grad_tiling_key.h"

using namespace AscendC;

template <uint64_t schId, uint64_t isInt32, uint64_t isCenter, uint64_t isDetermine>
__global__ __aicore__ void resize_linear_grad(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if constexpr (schId == 1) {
        // 纯datacopy模板，直接使用的simd的datacopy,不需要确定性
        ResizeLinearGrad::ResizeLinearGradTensorMove<DTYPE_GRADS> op(pipe);
        op.Init(grads, originalImage, y, &tilingData);
        op.Process();
        return;
    }
    if constexpr (schId == 3) {
        // outputL = 1 and alignCorners is True,无确定性问题，但是需要清零, 直接grads的nc搬出到输出y的nc上即可
        if constexpr (isInt32 == 1) {
            ResizeLinearGrad::ResizeLinearGradOut1<DTYPE_GRADS, uint32_t> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeLinearGrad::ResizeLinearGradOut1<DTYPE_GRADS, uint64_t> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }
    if constexpr (schId == 4) {
        // outputL = 1 and alignCorners is False,
        // 只有scaleL是正奇数倍才可以特殊处理，输入某个点直接搬出即可，无确定性问题
        if constexpr (isInt32 == 1) {
            ResizeLinearGrad::ResizeLinearGradOut1Half<DTYPE_GRADS, uint32_t> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeLinearGrad::ResizeLinearGradOut1Half<DTYPE_GRADS, uint64_t> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }
    if constexpr (isDetermine == 1) {
        // 确定性实现分支
        if constexpr (isInt32 == 1) {
            ResizeLinearGrad::ResizeLinearGradDetermine<DTYPE_GRADS, uint32_t, int32_t, isCenter, schId> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeLinearGrad::ResizeLinearGradDetermine<DTYPE_GRADS, uint64_t, int64_t, isCenter, schId> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
            return;
        }
    } else {
        // 非确定性实现分支
        if constexpr (isInt32 == 1) {
            ResizeLinearGrad::ResizeLinearGradNoDetermine<DTYPE_GRADS, uint32_t, isCenter, schId> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeLinearGrad::ResizeLinearGradNoDetermine<DTYPE_GRADS, uint64_t, isCenter, schId> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
            return;
        }
    }
}
