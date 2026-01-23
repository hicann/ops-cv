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
 * \file resize_bilinear_v2_base.h
 * \brief resize_bilinear_v2_base
 */
#ifndef RESIZE_BILINEAR_V2_BASE_H
#define RESIZE_BILINEAR_V2_BASE_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

constexpr int64_t ONE_BLOCK_BYTE = 32;
constexpr int64_t DIMS_THREE = 3;
constexpr int64_t DIMS_FOUR = 4;
constexpr int32_t MAX_DIM_NUM = 5;
constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t THREAD_NUM_MIDDLE = 512;
constexpr int32_t SIMT_SCENE_ALL_COPY = 1;
constexpr int32_t SIMT_SCENE_INPUT_ONE = 2;
constexpr int32_t SIMT_SCENE_OUTPUT_ONE = 3;

class ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2Base(){};

    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe)
    {
        this->pipe_ = pipe;

        this->xGM_.SetGlobalBuffer((__gm__ uint8_t*)x);
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

protected:
    TPipe* pipe_;

    GlobalTensor<uint8_t> xGM_;
    GlobalTensor<uint8_t> yGM_;
};

} // namespace ResizeBilinearV2
#endif // RESIZE_BILINEAR_V2_BASE_H
