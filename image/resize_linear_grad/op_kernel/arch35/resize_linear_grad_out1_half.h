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
 * \file resize_linear_grad_out1_half.h
 * \brief resize_linear_grad_out1_half
 */

#ifndef CANN_RESIZE_LINEAR_GRAD_OUT1_HALF_H
#define CANN_RESIZE_LINEAR_GRAD_OUT1_HALF_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_linear_grad_simt_base.h"

namespace ResizeLinearGrad {
using namespace AscendC;

// L of grads shape == 1 and align_corners = False and scaleL is odd numbers
template <typename T1, typename T2>
class ResizeLinearGradOut1Half {
public:
    __aicore__ inline ResizeLinearGradOut1Half(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
        const ResizeLinearGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    uint32_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
    const ResizeLinearGradTilingData *tilingData_;
};

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void SimtComputeMode4(T2 blkStartOffset, T2 blkProcessNum, T2 srcL,
    float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    T2 srcL1 = srcL - 1;
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 yGmIdx = blkStartOffset + idx;
        float srcIdsF = scaleL * 0.5f - 0.5f;
        T2 floorIds = Simt::Floor(srcIdsF);
        floorIds = Simt::Min(floorIds, srcL1);
        outputGm[yGmIdx * srcL + floorIds] = inputGm[yGmIdx];
    }
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(512) __aicore__ void calleeInt64O1Half(T2 blkStartOffset, T2 blkProcessNum, T2 srcL,
    float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeMode4<T1, T2>(blkStartOffset, blkProcessNum, srcL, scaleL, inputGm, outputGm);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ void calleeInt32O1Half(T2 blkStartOffset, T2 blkProcessNum, T2 srcL,
    float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeMode4<T1, T2>(blkStartOffset, blkProcessNum, srcL, scaleL, inputGm, outputGm);
}

template <typename T1, typename T2>
__aicore__ inline void ResizeLinearGradOut1Half<T1, T2>::Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
    const ResizeLinearGradTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1 *)grads);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
}

template <typename T1, typename T2>
__aicore__ inline void ResizeLinearGradOut1Half<T1, T2>::Process()
{
    T2 blkProcessNumC = tilingData_->ubFactor;
    T2 blkStartOffsetC = blockIdx_ * blkProcessNumC;
    if (blockIdx_ >= tilingData_->ubFactorTailB) {
        blkProcessNumC = tilingData_->ubFactorTailT;
        blkStartOffsetC = blockIdx_ * blkProcessNumC + tilingData_->ubFactorTailB;
    }
    T2 blkProcessNum = tilingData_->blkProcessNum;
    T2 blkStartOffset = blockIdx_ * blkProcessNum;
    if (blockIdx_ >= tilingData_->ubLoopSizeB) {
        blkProcessNum = tilingData_->ubLoopSizeT;
        blkStartOffset = blockIdx_ * blkProcessNum + tilingData_->ubLoopSizeB;
    }
    if (blockIdx_ < tilingData_->initCoreNum) {
        InitOutput<T1>(outputGm_[blkStartOffsetC], blkProcessNumC, T1(0));
    }
    SyncAll();

    if (blockIdx_ < tilingData_->realCoreNum) {
        float scaleL = tilingData_->scaleL;
        T2 srcL = (T2)(tilingData_->lenSrcLOrUb);
        if constexpr (sizeof(T2) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64O1Half<T1, T2>>(Simt::Dim3(512), blkStartOffset, blkProcessNum, srcL, scaleL,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        } else {
            Simt::VF_CALL<calleeInt32O1Half<T1, T2>>(Simt::Dim3(1024), blkStartOffset, blkProcessNum, srcL, scaleL,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        }
    }
}
} // namespace ResizeLinearGrad
#endif // CANN_RESIZE_LINEAR_GRAD_OUT1_HALF_H
