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
 * \file upsample_bilinear2d_common.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_COMMON_H
#define UPSAMPLE_BILINEAR2D_COMMON_H
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ADDR_ALIGN_SIZE = 128;

template <typename T1>
__aicore__ inline T1 Min(const T1 a, const T1 b)
{
    return a < b ? a : b;
};

template <typename T1>
__aicore__ inline T1 Max(const T1 a, const T1 b)
{
    return a > b ? a : b;
};

__aicore__ inline float getCenterValue(const int64_t dstIdx, const float scale, const bool align_corners)
{
    if (align_corners) {
        return scale * dstIdx;
    } else {
        float rel = scale * (dstIdx + (float)0.5) - (float)0.5;
        return Max(rel, (float)0.0);
    }
};

__aicore__ inline float getLambda(const float i_rel_idx, const int64_t i_min)
{
    float i_lambda = Min(Max(static_cast<float>(i_rel_idx - i_min), (float)0.0), (float)1.0);
    return i_lambda;
};

__aicore__ inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = 1e-6;
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

#endif  // UPSAMPLE_BILINEAR2D_COMMON_H