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
 * \file resize_linear_grad_simt_base.h
 * \brief resize_linear_grad_simt_base
 */
#ifndef RESIZE_LINEAR_GRAD_SIMT_BASE_H
#define RESIZE_LINEAR_GRAD_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ResizeLinearGrad {
using namespace AscendC;

template <typename T2, uint64_t isCenter>
__aicore__ __attribute__((always_inline)) inline float ComputeOriL(T2 L, float scaleL)
{
    // 调用Fma是为了规避编译器问题，不然后面算权重编译器自动优化成ffma，导致权重升精度，但是ori没有，从而某些情况会计算错误
    if constexpr (isCenter == 0) {
        float origWidth = Simt::Fma((static_cast<float>(L) + 0.5f), scaleL, -0.5f);
        float origWidthNew = Simt::Max(static_cast<float>(0.0f), origWidth);
        return origWidthNew;
    } else {
        float origWidth = Simt::Fma(static_cast<float>(L), scaleL, 0.0f);
        return origWidth;
    }
}

template <typename T2, uint64_t isCenter>
__aicore__ __attribute__((always_inline)) inline float ComputeOutL(T2 inLStart, float scaleL)
{
    if constexpr (isCenter == 1) {
        float outLStart = static_cast<float>(inLStart) * scaleL;
        return outLStart;
    } else {
        float outLStart = (static_cast<float>(inLStart) + 0.5f) * scaleL - 0.5f;
        return outLStart;
    }
}
} // namespace ResizeLinearGrad
#endif // RESIZE_NEAREAST_NEIGHBOR_V2_BASE_H
