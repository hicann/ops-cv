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
 * \file upsample_bicubic2d_aa_grad_common.h
 * \brief Common definitions for upsample_bicubic2d_aa_grad
 */

#ifndef UPSAMPLE_BICUBIC2D_AA_GRAD_COMMON_H
#define UPSAMPLE_BICUBIC2D_AA_GRAD_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_grad_tiling_data.h"

namespace UpsampleBicubic2dAAGrad {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 2048;
const int32_t THREAD_NUM_B64 = 512;
const uint64_t SCH_ID_1 = 1;

static __aicore__ inline float CubicConvolution1(float x)
{
    return static_cast<float>((1.5f * x - 2.5f) * x * x + 1.0f);
}
static __aicore__ inline float CubicConvolution2(float x)
{
    return static_cast<float>(((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f);
}
static __aicore__ inline float CubicFilterAA(float x)
{
    x = Simt::Abs(x);
    if (x < 1.0f) {
        return CubicConvolution1(x);
    } else if (x < 2.0f) {
        return CubicConvolution2(x);
    } else {
        return 0.0f;
    }
}
} // namespace UpsampleBicubic2dAAGrad

#endif // UPSAMPLE_BICUBIC2D_AA_GRAD_COMMON_H