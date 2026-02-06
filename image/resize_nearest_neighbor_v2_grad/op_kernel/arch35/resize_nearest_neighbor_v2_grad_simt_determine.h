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

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_H

#include "resize_nearest_neighbor_v2_grad_simt_base.h"

namespace ResizeNearestNeighborV2Grad
{
using namespace AscendC;

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
class ResizeNearestNeighborV2GradSimtDetermine : public ResizeNearestNeighborV2GradBase<T_DATA>
{
public:
    __aicore__ inline ResizeNearestNeighborV2GradSimtDetermine(){};
    __aicore__ inline void Process();
};

template <typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline void CalcOutputDimIdx(
    T_IDX yIdx, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW,
    T_IDX shiftW, T_IDX& idxN, T_IDX& idxC, T_IDX& yIdxH, T_IDX& yIdxW)
{
    T_IDX tmpIdx = yIdx;
    T_IDX tmpRes = 0;
    if constexpr (FORMAT == FORMAT_NCHW) {
        tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        yIdxW = tmpIdx - tmpRes * lenSrcW;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mH, shiftH);
        yIdxH = tmpIdx - tmpRes * lenSrcH;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mC, shiftC);
        idxC = tmpIdx - tmpRes * lenC;
        idxN = tmpRes;
    } else {
        tmpRes = Simt::UintDiv(tmpIdx, mC, shiftC);
        idxC = tmpIdx - tmpRes * lenC;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        yIdxW = tmpIdx - tmpRes * lenSrcW;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mH, shiftH);
        yIdxH = tmpIdx - tmpRes * lenSrcH;
        idxN = tmpRes;
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline float GetInputValue(
    __gm__ T_DATA* grads, T_IDX lenC, T_IDX lenDstH, T_IDX lenDstW, T_IDX idxN, T_IDX idxC, T_IDX gradsIdxH,
    T_IDX gradsIdxW)
{
    T_IDX gradsIdx = 0;
    if constexpr (FORMAT == FORMAT_NCHW) {
        gradsIdx = ((idxN * lenC + idxC) * lenDstH + gradsIdxH) * lenDstW + gradsIdxW;
    } else {
        gradsIdx = ((idxN * lenDstH + gradsIdxH) * lenDstW + gradsIdxW) * lenC + idxC;
    }

    float gradsValue = static_cast<float>(grads[gradsIdx]);
    return gradsValue;
}

template <typename T_IDX, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void CalGradsIdx(
    float scale, T_IDX yIdx, T_IDX gradsSize, T_IDX& idx)
{
    float offset = 0.0f;
    if constexpr (HALF_PIXEL) {
        offset = 0.5f;
    }
    idx = Simt::Min(static_cast<T_IDX>(Simt::Ceil(yIdx * scale - offset)), gradsSize);
}


template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtDetermineCompute(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float scaleH, float scaleW, float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mC,
    T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yIdx = coreOffset + idx;
        T_IDX idxN = 0, idxC = 0, yIdxH = 0, yIdxW = 0;
        CalcOutputDimIdx<T_IDX, FORMAT>(
            yIdx, lenC, lenSrcH, lenSrcW, mC, shiftC, mH, shiftH, mW, shiftW, idxN, idxC, yIdxH, yIdxW);

        T_IDX gradsIdxHStart = 0, gradsIdxWStart = 0;
        T_IDX gradsIdxHEnd = 0, gradsIdxWEnd = 0;
        float addValue = 0.0f;
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleH, yIdxH, lenDstH, gradsIdxHStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleH, yIdxH + 1, lenDstH, gradsIdxHEnd);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW, lenDstW, gradsIdxWStart);
        CalGradsIdx<T_IDX, HALF_PIXEL>(inverseScaleW, yIdxW + 1, lenDstW, gradsIdxWEnd);

        for (T_IDX i = gradsIdxHStart; i < gradsIdxHEnd; i++) {
            #pragma unroll
            for (T_IDX j = gradsIdxWStart; j < gradsIdxWEnd; j++) {
                addValue += GetInputValue<T_DATA, T_IDX, FORMAT>(
                    grads, lenC, lenDstH, lenDstW, idxN, idxC, i, j);
            }
        }

        y[yIdx] = static_cast<T_DATA>(addValue);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT32) __aicore__ void calleeSimtDetermineInt32(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float scaleH, float scaleW, float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mC,
    T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineCompute<T_DATA, T_IDX, FORMAT, HALF_PIXEL>(
        grads, y, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, inverseScaleH, inverseScaleW, coreFactor,
        coreOffset, mC, shiftC, mH, shiftH, mW, shiftW);
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT64) __aicore__ void calleeSimtDetermineInt64(
    __gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW,
    float scaleH, float scaleW, float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mC,
    T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineCompute<T_DATA, T_IDX, FORMAT, HALF_PIXEL>(
        grads, y, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, inverseScaleH, inverseScaleW, coreFactor,
        coreOffset, mC, shiftC, mH, shiftH, mW, shiftW);
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool HALF_PIXEL>
__aicore__ inline void ResizeNearestNeighborV2GradSimtDetermine<T_DATA, T_IDX, FORMAT, HALF_PIXEL>::Process()
{
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
    
    T_IDX mC = 1, mH = 1, mW = 1;
    T_IDX shiftC = 1, shiftH = 1, shiftW = 1;
    GetUintDivMagicAndShift(mC, shiftC, lenC);
    GetUintDivMagicAndShift(mH, shiftH, lenSrcH);
    GetUintDivMagicAndShift(mW, shiftW, lenSrcW);

    if (this->blockIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtDetermineInt32<T_DATA, T_IDX, FORMAT, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT32), (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW,
                inverseScaleH, inverseScaleW, blkProcessNum, blkStartOffset, mC, shiftC, mH, shiftH, mW, shiftW);
        } else {
            Simt::VF_CALL<calleeSimtDetermineInt64<T_DATA, T_IDX, FORMAT, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT64), (__gm__ T_DATA*)(this->gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(this->yGm_.GetPhyAddr()), lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW,
                inverseScaleH, inverseScaleW, blkProcessNum, blkStartOffset, mC, shiftC, mH, shiftH, mW, shiftW);
        }
    }
}
}  // namespace ResizeNearestNeighborV2Grad

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_DETERMINE_H