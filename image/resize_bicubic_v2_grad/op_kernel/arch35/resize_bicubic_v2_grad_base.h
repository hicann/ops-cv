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
 * \file resize_bicubic_v2_grad_base.h
 * \brief resize_bicubic_v2_grad_base
 */

#ifndef CANN_RESIZE_BICUBIC_V2_GRAD_BASE_H
#define CANN_RESIZE_BICUBIC_V2_GRAD_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ResizeBicubicV2Grad {
using namespace AscendC;

constexpr int64_t DB_BUFFER_NUM = 2;
constexpr uint32_t INDEX_NUM_ZERO = 0;
constexpr uint32_t INDEX_NUM_ONE = 1;
constexpr uint32_t INDEX_NUM_TWO = 2;
constexpr uint32_t INDEX_NUM_THREE = 3;

__aicore__ inline int64_t Min(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

__aicore__ __attribute__((always_inline)) inline float CalcCubicConvolution1(float t, float a)
{
    return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
}

__aicore__ __attribute__((always_inline)) inline float CalcCubicConvolution2(float t, float a)
{
    return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
}

__aicore__ __attribute__((always_inline)) inline float CalcCubicCoefficient(float idxDiff)
{
    float a = -0.75f;
    float coeff = 0.0f;

    if (idxDiff <= 1.0f) {
        coeff = CalcCubicConvolution1(idxDiff, a);
    } else if (idxDiff < 2.0f) {
        coeff = CalcCubicConvolution2(idxDiff, a);
    }

    return coeff;
}

__aicore__ __attribute__((always_inline)) inline void CalcCubicCoefficients(float coeffs[4], float idxDiff)
{
    float a = -0.75f;

    float t1 = idxDiff;
    coeffs[INDEX_NUM_ZERO] = CalcCubicConvolution2(t1 + 1.0f, a);
    coeffs[INDEX_NUM_ONE] = CalcCubicConvolution1(t1, a);

    float t2 = 1.0f - idxDiff;
    coeffs[INDEX_NUM_TWO] = CalcCubicConvolution1(t2, a);
    coeffs[INDEX_NUM_THREE] = CalcCubicConvolution2(t2 + 1.0f, a);
}

template <typename T_IDX, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline float CalcSourceIndex(float scale, T_IDX dstIdx)
{
    if constexpr (ALIGN_CORNERS) {
        return scale * static_cast<float>(dstIdx);
    } else {
        return scale * (static_cast<float>(dstIdx) + 0.5f) - 0.5f;
    }
}

}  // namespace ResizeBicubicV2Grad

#endif  // CANN_RESIZE_BICUBIC_V2_GRAD_BASE_H
