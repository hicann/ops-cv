/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2_grad_simt_base.h
 * \brief resize_nearest_neighbor_v2_grad_simt_base
 */
 
#ifndef RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_BASE_H
#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_BASE_H

#include "kernel_operator.h"

namespace ResizeNearestNeighborV2Grad {
using namespace AscendC;

constexpr int32_t MAX_DIM_NUM = 5;
constexpr int32_t SIMT_THREAD_NUM_INT32 = 2048;//2048 & 1024 对比下
constexpr int32_t SIMT_THREAD_NUM_INT64 = 512;//1024
constexpr float HALF_PIXEL_VAL = 0.5f;
constexpr int32_t BIT64 = 64;
constexpr int32_t BIT32 = 32;


template <typename T_DATA>
class ResizeNearestNeighborV2GradBase {
public:
    __aicore__ inline ResizeNearestNeighborV2GradBase(){};
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR y, const ResizeNearestNeighborV2GradTilingData* tilingData);
protected:
    GlobalTensor<T_DATA> gradsGm_;
    GlobalTensor<T_DATA> yGm_;
    int32_t blockIdx_;
    const ResizeNearestNeighborV2GradTilingData* tilingData_;
};

template <typename T_DATA>
__aicore__ inline void ResizeNearestNeighborV2GradBase<T_DATA>::Init(
    GM_ADDR grads, GM_ADDR y, const ResizeNearestNeighborV2GradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    gradsGm_.SetGlobalBuffer((__gm__ T_DATA*)grads); 
    yGm_.SetGlobalBuffer((__gm__ T_DATA*)y);
}

}  // namespace ResizeNearestNeighborV2GradBase

#endif
