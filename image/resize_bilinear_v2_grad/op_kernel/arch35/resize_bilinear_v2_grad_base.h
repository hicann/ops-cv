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
 * \file resize_bilinear_v2_grad_base.h
 * \brief resize_bilinear_v2_grad_base
 */
#ifndef RESIZE_BILINEAR_V2_GRAD_BASE_H
#define RESIZE_BILINEAR_V2_GRAD_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2Grad {
using namespace AscendC;

constexpr int64_t ONE_BLOCK_BYTE = 32;
constexpr int64_t DIMS_THREE = 3;
constexpr int64_t DIMS_FOUR = 4;

constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t THREAD_NUM = 512;
constexpr int32_t THREAD_NUM_MIDDLE = 256;

template <typename T_IDX>
__aicore__ __attribute__((always_inline)) inline void QuickDivForSimtComputeDetermine(
    T_IDX& tmp, T_IDX& mW, T_IDX shiftW, T_IDX& W, T_IDX lenSrcW, T_IDX& mH, T_IDX shiftH, T_IDX& H, T_IDX lenSrcH,
    T_IDX& mC, T_IDX shiftC, T_IDX& C, T_IDX lenC, T_IDX& mN, T_IDX shiftN, T_IDX& N, T_IDX lenN)
{
    T_IDX tmpRes = Simt::UintDiv(tmp, mW, shiftW);
    W = tmp - tmpRes * lenSrcW;
    tmp = tmpRes;

    tmpRes = Simt::UintDiv(tmp, mH, shiftH);
    H = tmp - tmpRes * lenSrcH;
    tmp = tmpRes;

    tmpRes = Simt::UintDiv(tmp, mC, shiftC);
    C = tmp - tmpRes * lenC;
    tmp = tmpRes;

    tmpRes = Simt::UintDiv(tmp, mN, shiftN);
    N = tmp - tmpRes * lenN;
}

class ResizeBilinearV2GradBase {
public:
    __aicore__ inline ResizeBilinearV2GradBase(){};

    __aicore__ inline void BaseInit(GM_ADDR grads, GM_ADDR y, TPipe* pipe)
    {
        this->pipe_ = pipe;

        this->gradsGM_.SetGlobalBuffer((__gm__ uint8_t*)grads);
        this->yGM_.SetGlobalBuffer((__gm__ uint8_t*)y);
    };

protected:
    /*
     * Ceil operation
     */
    __aicore__ inline int64_t Ceil(float x)
    {
        return static_cast<int64_t>(x) + 1;
    }

    /*
     * Floor operation
     */
    __aicore__ inline int64_t Floor(float x)
    {
        return static_cast<int64_t>(x);
    }

    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return (a < b) ? a : b;
    }

    __aicore__ inline int64_t Max(int64_t a, int64_t b)
    {
        return (a > b) ? a : b;
    }

protected:
    TPipe* pipe_;

    GlobalTensor<uint8_t> gradsGM_;
    GlobalTensor<uint8_t> yGM_;
};

} // namespace ResizeBilinearV2Grad
#endif // RESIZE_BILINEAR_V2_GRAD_BASE_H
