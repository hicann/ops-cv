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
 * \file resize_nearest_neighbor_v2_grad_simt_determine_hw.h
 * \brief ResizeNearestNeighborV2Grad simt determine hw
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_HW_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_HW_H

#include "resize_nearest_neighbor_v2_grad_simt_base.h"

namespace ResizeNearestNeighborV2Grad
{
using namespace AscendC;

constexpr bool IS_UNROLL_W = true;

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtDetermineHWCompute(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
        idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yIdx = coreOffset + idx;

        // calculate the index of y in H and W dim.
        T_IDX yIdxH = 0;
        T_IDX yIdxW = 0;
        T_IDX tmpIdx = yIdx;
        T_IDX tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        yIdxW = tmpIdx - tmpRes * lenSrcW;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mH, shiftH);
        yIdxH = tmpIdx - tmpRes * lenSrcH;
        
        T_IDX gradsIdxHStart = 0;
        T_IDX gradsIdxWStart = 0;
        T_IDX gradsIdxHEnd = 0;
        T_IDX gradsIdxWEnd = 0;
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleH, yIdxH, lenDstH, gradsIdxHStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleH, yIdxH + 1, lenDstH, gradsIdxHEnd);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW, lenDstW, gradsIdxWStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW + 1, lenDstW, gradsIdxWEnd);

        T_IDX lenSrcHW = lenSrcH * lenSrcW;
        T_IDX lenDstHW = lenDstH * lenDstW;

        for (T_IDX n = 0; n < lenN; n++) {
            for (T_IDX c = 0; c < lenC; c++) {
                T_IDX gradsBaseIdx = n * (lenC * lenDstHW) + c * lenDstHW;
                T_IDX yTempIdx = n * (lenC * lenSrcHW) + c * lenSrcHW + yIdx;
                float addValue = 0.0f;
                for (T_IDX i = gradsIdxHStart; i < gradsIdxHEnd; i++) {
                    #pragma unroll
                    for (T_IDX j = gradsIdxWStart; j < gradsIdxWEnd; j++) {
                        addValue += static_cast<float>(grads[gradsBaseIdx + i * lenDstW + j]);
                    }
                }
                y[yTempIdx] = static_cast<T_DATA>(addValue);
            }
        }
    }
}

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT32) __aicore__ void calleeSimtDetermineHWInt32(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineHWCompute<T_DATA, T_IDX, HALF_PIXEL>(
        grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, inverseScaleH, inverseScaleW, coreFactor,
        coreOffset, mH, shiftH, mW, shiftW);
}

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT64) __aicore__ void calleeSimtDetermineHWInt64(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineHWCompute<T_DATA, T_IDX, HALF_PIXEL>(
        grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, inverseScaleH, inverseScaleW, coreFactor,
        coreOffset, mH, shiftH, mW, shiftW);
}

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
class ResizeNearestNeighborV2GradSimtDetermineHW : public ResizeNearestNeighborV2GradBase<T_DATA>
{
public:
    __aicore__ inline ResizeNearestNeighborV2GradSimtDetermineHW(){};
    __aicore__ inline void Process();
};


template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__aicore__ inline void ResizeNearestNeighborV2GradSimtDetermineHW<T_DATA, T_IDX, HALF_PIXEL>::Process()
{
    const T_IDX lenN = this->tilingData_->lenN;
    const T_IDX lenC = this->tilingData_->lenC;
    const T_IDX lenSrcH = this->tilingData_->lenSrcH;
    const T_IDX lenSrcW = this->tilingData_->lenSrcW;
    const T_IDX lenDstH = this->tilingData_->lenDstH;
    const T_IDX lenDstW = this->tilingData_->lenDstW;
    const int32_t useCoreNum = this->tilingData_->realCoreNum;
    const float scaleH = this->tilingData_->scaleH;
    const float scaleW = this->tilingData_->scaleW;
    const float inverseScaleH = this->tilingData_->inverseScaleH;
    const float inverseScaleW = this->tilingData_->inverseScaleW;

    T_IDX blkProcessNum = this->tilingData_->splitBlockFactor;
    T_IDX blkStartOffset = this->blockIdx_ * this->tilingData_->splitBlockFactor;
    if (this->blockIdx_ < this->tilingData_->splitBlockTailFactor) {
        blkProcessNum += 1;
        blkStartOffset += this->blockIdx_;
    } else {
        blkStartOffset += this->tilingData_->splitBlockTailFactor;
    }
    
    T_IDX mC = 1;
    T_IDX mH = 1;
    T_IDX mW = 1;
    T_IDX shiftC = 1;
    T_IDX shiftH = 1;
    T_IDX shiftW = 1;
    GetUintDivMagicAndShift(mH, shiftH, lenSrcH);
    GetUintDivMagicAndShift(mW, shiftW, lenSrcW);

    if (this->blockIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtDetermineHWInt32<T_DATA, T_IDX, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT32), (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, inverseScaleH, inverseScaleW,
                blkProcessNum, blkStartOffset, mH, shiftH, mW, shiftW);
        } else {
            Simt::VF_CALL<calleeSimtDetermineHWInt64<T_DATA, T_IDX, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT64), (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, inverseScaleH, inverseScaleW,
                blkProcessNum, blkStartOffset, mH, shiftH, mW, shiftW);
        }
    }
}

} // namespace ResizeNearestNeighborV2Grad
#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_HW_H

