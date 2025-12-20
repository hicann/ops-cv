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
 * \file resize_linear_grad_simt_determine.h
 * \brief resize_linear_grad_simt_determine
 */

#ifndef CANN_RESIZE_LINEAR_GRAD_DETERMINE_H
#define CANN_RESIZE_LINEAR_GRAD_DETERMINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ResizeLinearGrad {
using namespace AscendC;

template <typename T1, typename T2, typename T3, uint64_t isCenter, uint64_t mode>
class ResizeLinearGradDetermine {
public:
    __aicore__ inline ResizeLinearGradDetermine(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
        const ResizeLinearGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    uint32_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
    const ResizeLinearGradTilingData *tilingData_;
};

template <typename T1, typename T2, uint64_t isCenter>
__aicore__ __attribute__((always_inline)) inline void ComputeLimt(T2 inLStart, T2 lenDesL, T2 srcL1, T2 NC, T2 L,
    float outLStart, float scaleL, float &acc, __gm__ T1 *inputGm)
{
    while (inLStart < lenDesL) {
        float posOutL = Simt::Max(0.0f, outLStart);
        T2 LFloorIndex = static_cast<T2>(Simt::Floor(posOutL));
        T2 LTempIndex = Simt::Min(LFloorIndex, srcL1);
        float lep[2];
        lep[0] = 1.0f;
        lep[1] = posOutL - static_cast<float>(LTempIndex);
        T2 inIds = NC * lenDesL + inLStart;
        float inputData = static_cast<float>(inputGm[inIds]);
        for (int32_t i = 0; i < 2; i++) {
            if (LTempIndex + i == L) {
                acc += inputData * lep[i];
            }
        }
        inLStart++;
        outLStart = ComputeOutL<T2, isCenter>(inLStart, scaleL);
    }
}

template <typename T1, typename T2, typename T3, uint64_t isCenter>
__aicore__ __attribute__((always_inline)) inline void SimtComputeDetermine(T2 blkStartOffset, T2 blkProcessNum, T2 mL,
    T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb, float scaleL, float inverseScaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    T2 srcL1 = lenSrcLOrUb - 1;
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 NC = Simt::UintDiv(yGmIdx, mL, shiftL);
        T2 L = yGmIdx - NC * lenSrcLOrUb;
        T2 inLStart = 0;
        float outLStart = 0.0f;
        float acc = 0.0f;
        if constexpr (isCenter == 1) {
            inLStart = static_cast<T2>(Simt::Max(static_cast<T3>(0),
                static_cast<T3>(Simt::Ceil((static_cast<float>(L) - 1.0f) * inverseScaleL))));
            outLStart = static_cast<float>(inLStart) * scaleL;
        } else {
            inLStart = static_cast<T2>(Simt::Max(static_cast<T3>(0),
                static_cast<T3>(Simt::Ceil((static_cast<float>(L) - 1.0f + 0.5f) * inverseScaleL - 0.5f))));
            outLStart = (static_cast<float>(inLStart) + 0.5f) * scaleL - 0.5f;
        }
        float limetL = static_cast<float>(L) + 1.0f;
        if (L != srcL1) {
            while (outLStart < limetL && inLStart < lenDesL) {
                float posOutL = Simt::Max(0.0f, outLStart);
                float lep = 1.0f - Simt::Abs(posOutL - static_cast<float>(L));
                T2 inIdx = NC * lenDesL + inLStart;
                float inputValue = static_cast<float>(inputGm[inIdx]);
                acc += inputValue * lep;
                inLStart++;
                outLStart = ComputeOutL<T2, isCenter>(inLStart, scaleL);
            }
        } else {
            ComputeLimt<T1, T2, isCenter>(inLStart, lenDesL, srcL1, NC, L, outLStart, scaleL, acc, inputGm);
        }
        outputGm[yGmIdx] = static_cast<T1>(acc);
    }
}

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void SimtComputeDetermineMode2(T2 blkStartOffset, T2 blkProcessNum,
    T2 lenDesL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 NC = blkStartOffset + idx; // SRC_L=1, 就是reducesum操作
        float val = 0.0f;
        for (T2 l = 0; l < lenDesL; l++) {
            float inputData = static_cast<float>(inputGm[NC * lenDesL + l]);
            val += inputData;
        }
        outputGm[NC] = static_cast<T1>(val);
    }
}

template <typename T1, typename T2, typename T3, uint64_t isCenter>
__simt_vf__ LAUNCH_BOUND(512) __aicore__ void calleeInt64Determine(T2 blkStartOffset, T2 blkProcessNum, T2 mL,
    T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb, float scaleL, float inverseScaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeDetermine<T1, T2, T3, isCenter>(blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
        inverseScaleL, inputGm, outputGm);
}

template <typename T1, typename T2, typename T3, uint64_t isCenter>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ void calleeInt32Determine(T2 blkStartOffset, T2 blkProcessNum, T2 mL,
    T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb, float scaleL, float inverseScaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeDetermine<T1, T2, T3, isCenter>(blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
        inverseScaleL, inputGm, outputGm);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(512) __aicore__
    void calleeInt64Mode2(T2 blkStartOffset, T2 blkProcessNum, T2 lenDesL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeDetermineMode2<T1, T2>(blkStartOffset, blkProcessNum, lenDesL, inputGm, outputGm);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__
    void calleeInt32Mode2(T2 blkStartOffset, T2 blkProcessNum, T2 lenDesL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtComputeDetermineMode2<T1, T2>(blkStartOffset, blkProcessNum, lenDesL, inputGm, outputGm);
}

template <typename T1, typename T2, typename T3, uint64_t isCenter, uint64_t mode>
__aicore__ inline void ResizeLinearGradDetermine<T1, T2, T3, isCenter, mode>::Init(GM_ADDR grads, GM_ADDR originalImage,
    GM_ADDR y, const ResizeLinearGradTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1 *)grads);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
}

template <typename T1, typename T2, typename T3, uint64_t isCenter, uint64_t mode>
__aicore__ inline void ResizeLinearGradDetermine<T1, T2, T3, isCenter, mode>::Process()
{
    if (blockIdx_ > tilingData_->realCoreNum - 1) {
        return;
    }
    T2 blkProcessNum = tilingData_->blkProcessNum;
    T2 blkStartOffset = blockIdx_ * blkProcessNum;
    if (blockIdx_ >= tilingData_->ubLoopSizeB) {
        blkProcessNum = tilingData_->ubLoopSizeT;
        blkStartOffset = blockIdx_ * blkProcessNum + tilingData_->ubLoopSizeB;
    }
    T2 lenDesL = (T2)(tilingData_->lenDesL);
    if constexpr (mode == 2) {
        if constexpr (sizeof(T2) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64Mode2<T1, T2>>(Simt::Dim3(512), blkStartOffset, blkProcessNum, lenDesL,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        } else {
            Simt::VF_CALL<calleeInt32Mode2<T1, T2>>(Simt::Dim3(1024), blkStartOffset, blkProcessNum, lenDesL,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        }
    } else {
        T2 mL = 0;
        T2 shiftL = 0;
        T2 lenSrcLOrUb = (T2)(tilingData_->lenSrcLOrUb);
        GetUintDivMagicAndShift(mL, shiftL, lenSrcLOrUb);

        float scaleL = tilingData_->scaleL;
        float inverseScaleL = tilingData_->inverseScaleL;
        if constexpr (sizeof(T2) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64Determine<T1, T2, T3, isCenter>>(Simt::Dim3(512), blkStartOffset, blkProcessNum,
                mL, shiftL, lenDesL, lenSrcLOrUb, scaleL, inverseScaleL, (__gm__ T1 *)(inputGm_.GetPhyAddr()),
                (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        } else {
            Simt::VF_CALL<calleeInt32Determine<T1, T2, T3, isCenter>>(Simt::Dim3(1024), blkStartOffset, blkProcessNum,
                mL, shiftL, lenDesL, lenSrcLOrUb, scaleL, inverseScaleL, (__gm__ T1 *)(inputGm_.GetPhyAddr()),
                (__gm__ T1 *)(outputGm_.GetPhyAddr()));
        }
    }
}
} // namespace ResizeLinearGrad
#endif // CANN_RESIZE_LINEAR_GRAD_DETERMINE_H
