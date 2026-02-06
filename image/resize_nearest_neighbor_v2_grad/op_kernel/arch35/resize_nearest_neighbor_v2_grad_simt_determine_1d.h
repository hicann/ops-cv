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
 * \file resize_nearest_neighbor_v2_grad_simt_determine.h
 * \brief resize_nearest_neighbor_v2_grad_simt_determine
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_1D_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_1D_H

#include "resize_nearest_neighbor_v2_grad_simt_base.h"

namespace ResizeNearestNeighborV2Grad
{
using namespace AscendC;

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtDetermine1DNCHWCompute(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcW, T_IDX lenDstW, float inverseScaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
        idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yIdx = coreOffset + idx;

        T_IDX idxNC = 0;
        T_IDX yIdxW = 0;
        T_IDX tmpIdx = yIdx;
        idxNC = Simt::UintDiv(tmpIdx, mW, shiftW);
        yIdxW = tmpIdx - idxNC * lenSrcW;

        T_IDX gradsIdxWStart = 0;
        T_IDX gradsIdxWEnd = 0;
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW, lenDstW, gradsIdxWStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW + 1, lenDstW, gradsIdxWEnd);

        float addValue = 0; // consider add in gm need more time.
        for (T_IDX w = gradsIdxWStart; w < gradsIdxWEnd; w++) {
            addValue += static_cast<float>(grads[idxNC * lenDstW + w]);
        }
        y[yIdx] = static_cast<T_DATA>(addValue);
    }
}

template <typename T_DATA, typename T_IDX, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtDetermine1DNHWCCompute(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcW, T_IDX lenDstW, float inverseScaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
        idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yIdx = coreOffset + idx;

        T_IDX idxN = 0;
        T_IDX idxC = 0; 
        T_IDX yIdxW = 0;
        T_IDX tmpIdx = yIdx;
        T_IDX tmpRes = Simt::UintDiv(tmpIdx, mC, shiftC);
        idxC = tmpIdx - tmpRes * lenC;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        yIdxW = tmpIdx - tmpRes * lenSrcW;
        idxN = tmpRes;

        T_IDX gradsIdxWStart = 0;
        T_IDX gradsIdxWEnd = 0;
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW, lenDstW, gradsIdxWStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW + 1, lenDstW, gradsIdxWEnd);

        float addValue = 0.0f; // consider add in gm need more time.
        for (T_IDX w = gradsIdxWStart; w < gradsIdxWEnd; w++) {
            addValue += static_cast<float>(grads[(idxN * lenDstW + w) * lenC + idxC]);
        }
        y[yIdx] = static_cast<T_DATA>(addValue);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT32) __aicore__ void calleeSimtDetermine1DInt32(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcW, T_IDX lenDstW, float inverseScaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mW, T_IDX shiftW)
{
    if constexpr (FORMAT == FORMAT_NCHW){
        SimtDetermine1DNCHWCompute<T_DATA, T_IDX, HALF_PIXEL>(
            grads, y, lenC, lenSrcW, lenDstW, inverseScaleW, coreFactor, coreOffset, mC, shiftC, mW, shiftW);
    }else {
        SimtDetermine1DNHWCCompute<T_DATA, T_IDX, HALF_PIXEL>(
            grads, y, lenC, lenSrcW, lenDstW, inverseScaleW, coreFactor, coreOffset, mC, shiftC, mW, shiftW);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT64) __aicore__ void calleeSimtDetermine1DInt64(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcW, T_IDX lenDstW, float inverseScaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mW, T_IDX shiftW)
{
    if constexpr (FORMAT == FORMAT_NCHW){
        SimtDetermine1DNCHWCompute<T_DATA, T_IDX, HALF_PIXEL>(
            grads, y, lenC, lenSrcW, lenDstW, inverseScaleW, coreFactor, coreOffset, mC, shiftC, mW, shiftW);
    }else {
        SimtDetermine1DNHWCCompute<T_DATA, T_IDX, HALF_PIXEL>(
            grads, y, lenC, lenSrcW, lenDstW, inverseScaleW, coreFactor, coreOffset, mC, shiftC, mW, shiftW);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
class ResizeNearestNeighborV2GradSimtDetermine1D : public ResizeNearestNeighborV2GradBase<T_DATA>
{
public:
    __aicore__ inline ResizeNearestNeighborV2GradSimtDetermine1D(){};
    __aicore__ inline void Process();
};


template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__aicore__ inline void ResizeNearestNeighborV2GradSimtDetermine1D<T_DATA, T_IDX, FORMAT, HALF_PIXEL>::Process()
{
    const T_IDX lenC = this->tilingData_->lenC;
    const T_IDX lenSrcW = this->tilingData_->lenSrcW;
    const T_IDX lenDstW = this->tilingData_->lenDstW;
    const int32_t useCoreNum = this->tilingData_->realCoreNum;
    const float scaleW = this->tilingData_->scaleW;
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
    T_IDX mW = 1;
    T_IDX shiftC = 1;
    T_IDX shiftW = 1;
    GetUintDivMagicAndShift(mC, shiftC, lenC);
    GetUintDivMagicAndShift(mW, shiftW, lenSrcW);

    if (this->blockIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtDetermine1DInt32<T_DATA, T_IDX, FORMAT, HALF_PIXEL>>(Simt::Dim3(SIMT_THREAD_NUM_INT32),
            (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()), (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenC, lenSrcW, lenDstW,
            inverseScaleW, blkProcessNum, blkStartOffset, mC, shiftC, mW, shiftW);
        } else {
            Simt::VF_CALL<calleeSimtDetermine1DInt64<T_DATA, T_IDX, FORMAT, HALF_PIXEL>>(Simt::Dim3(SIMT_THREAD_NUM_INT64),
            (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()), (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenC, lenSrcW, lenDstW,
            inverseScaleW, blkProcessNum, blkStartOffset, mC, shiftC, mW, shiftW);
        }
    }
}
}  // namespace ResizeNearestNeighborV2Grad

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_1D_H