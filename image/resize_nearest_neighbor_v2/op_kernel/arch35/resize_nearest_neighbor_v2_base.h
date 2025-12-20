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
 * \file resize_nearest_neighbor_v2_base.h
 * \brief resize_nearest_neighbor_v2_base
 */
#ifndef RESIZE_NEAREAST_NEIGHBOR_V2_BASE_H
#define RESIZE_NEAREAST_NEIGHBOR_V2_BASE_H

#include "../inc/platform.h"
#include "kernel_operator.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

constexpr int64_t DIMS_THREE = 3;
constexpr int64_t DIMS_FOUR = 4;
constexpr int32_t MAX_DIM_NUM = 5;
constexpr int32_t BIT64 = 64;
constexpr int32_t BIT32 = 32;
class ResizeNearestNeighborV2Base {
public:
    __aicore__ inline ResizeNearestNeighborV2Base(){};

protected:
    /*
     * Ceil operation
     */
    __aicore__ inline int32_t Ceil(float x)
    {
        return static_cast<int64_t>(static_cast<int32_t>(x) + 1);
    }

    /*
     * Floor operation
     */
    __aicore__ inline int32_t Floor(float x)
    {
        return static_cast<int64_t>(static_cast<int32_t>(x));
    }

    /*
     * Round operation
     */
    __aicore__ inline int32_t Round(float x)
    {
        return static_cast<int64_t>(static_cast<int32_t>(x + 0.5f));
    }

    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return (a < b) ? a : b;
    }

protected:
    int64_t srcHFactor_ = 0;
    int64_t srcWFactor_ = 0;

    // tiling接收的参数
    float hScale_ = 1.0f;  // HScale: srcH / dstH
    float wScale_ = 1.0f;
    int64_t srcHSize_ = 0;
    int64_t srcWSize_ = 0;
    int64_t dstHSize_ = 0;
    int64_t dstWSize_ = 0;
    int64_t hFactor_ = 0;
    int64_t wFactor_ = 0;
    int64_t hTailFactor_ = 0;
    int64_t wTailFactor_ = 0;
    int64_t hwCnt_ = 0;
    float bias_ = 0.0f;
    int64_t lenC = 0;
};

}  // namespace ResizeNearestNeighborV2

#endif  // RESIZE_NEAREAST_NEIGHBOR_V2_BASE_H
