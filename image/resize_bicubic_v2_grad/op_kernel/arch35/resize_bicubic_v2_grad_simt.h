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
 * \file resize_bicubic_v2_grad_simt.h
 * \brief resize_bicubic_v2_grad_simt
 */

#ifndef CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_H
#define CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_H

#include "resize_bicubic_v2_grad_base.h"

namespace ResizeBicubicV2Grad {
using namespace AscendC;

constexpr int32_t SIMT_THREAD_NUM_INT32 = 1024;
constexpr int32_t SIMT_THREAD_NUM_INT64 = 512;

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
class ResizeBicubicV2GradSimt {
public:
    __aicore__ inline ResizeBicubicV2GradSimt(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR y, const ResizeBicubicV2GradSimtTilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T_DATA> gradsGm_;
    GlobalTensor<T_DATA> yGm_;
    int32_t coreIdx_;
    const ResizeBicubicV2GradSimtTilingData *tilingData_;
};

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ inline void ResizeBicubicV2GradSimt<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>::Init(
    GM_ADDR grads, GM_ADDR y, const ResizeBicubicV2GradSimtTilingData *tilingData)
{
    coreIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    gradsGm_.SetGlobalBuffer((__gm__ T_DATA *)grads);
    yGm_.SetGlobalBuffer((__gm__ T_DATA *)y);
}

template <typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline void CalcInputDimIdx(T_IDX gradsIdx, T_IDX lenC, T_IDX lenDstH,
    T_IDX lenDstW, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW, T_IDX &idxN, T_IDX &idxC,
    T_IDX &gradsIdxH, T_IDX &gradsIdxW)
{
    T_IDX tmpIdx = gradsIdx;
    T_IDX tmpRes = 0;
    if constexpr (FORMAT == FORMAT_NCHW) {
        tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        gradsIdxW = tmpIdx - tmpRes * lenDstW;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mH, shiftH);
        gradsIdxH = tmpIdx - tmpRes * lenDstH;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mC, shiftC);
        idxC = tmpIdx - tmpRes * lenC;
        idxN = tmpRes;
    } else {
        tmpRes = Simt::UintDiv(tmpIdx, mC, shiftC);
        idxC = tmpIdx - tmpRes * lenC;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mW, shiftW);
        gradsIdxW = tmpIdx - tmpRes * lenDstW;
        tmpIdx = tmpRes;
        tmpRes = Simt::UintDiv(tmpIdx, mH, shiftH);
        gradsIdxH = tmpIdx - tmpRes * lenDstH;
        idxN = tmpRes;
    }
}

template <typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline T_IDX CalcOutputIdx(
    T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX idxN, T_IDX idxC, T_IDX idxH, T_IDX idxW)
{
    if constexpr (FORMAT == FORMAT_NCHW) {
        return ((idxN * lenC + idxC) * lenSrcH + idxH) * lenSrcW + idxW;
    } else {
        return ((idxN * lenSrcH + idxH) * lenSrcW + idxW) * lenC + idxC;
    }
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline void CalcOutputValue(__gm__ T_DATA *y, T_IDX lenC, T_IDX lenSrcH,
    T_IDX lenSrcW, T_IDX idxN, T_IDX idxC, T_IDX2 yIdxH, T_IDX2 yIdxW, float addValue)
{
    T_IDX yFinalIdxH =
        static_cast<T_IDX>(Simt::Max(static_cast<T_IDX2>(0), Simt::Min(yIdxH, static_cast<T_IDX2>(lenSrcH) - 1)));
    T_IDX yFinalIdxW =
        static_cast<T_IDX>(Simt::Max(static_cast<T_IDX2>(0), Simt::Min(yIdxW, static_cast<T_IDX2>(lenSrcW) - 1)));
    T_IDX yIdx = CalcOutputIdx<T_IDX, FORMAT>(lenC, lenSrcH, lenSrcW, idxN, idxC, yFinalIdxH, yFinalIdxW);
    Simt::AtomicAdd(y + yIdx, static_cast<T_DATA>(addValue));
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(__gm__ T_DATA *grads, __gm__ T_DATA *y, T_IDX lenC,
    T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX coreFactor,
    T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX gradsIdx = coreOffset + idx;
        T_IDX idxN = 0, idxC = 0, gradsIdxH = 0, gradsIdxW = 0;
        CalcInputDimIdx<T_IDX, FORMAT>(
            gradsIdx, lenC, lenDstH, lenDstW, mC, shiftC, mH, shiftH, mW, shiftW, idxN, idxC, gradsIdxH, gradsIdxW);

        float yRealIdxH = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleH, gradsIdxH);
        T_IDX2 yFloorIdxH = static_cast<T_IDX2>(Simt::Floor(yRealIdxH));
        float yIdxDiffH = yRealIdxH - static_cast<float>(yFloorIdxH);

        float yRealIdxW = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleW, gradsIdxW);
        T_IDX2 yFloorIdxW = static_cast<T_IDX2>(Simt::Floor(yRealIdxW));
        float yIdxDiffW = yRealIdxW - static_cast<float>(yFloorIdxW);

        float yCoeffsH[4];
        CalcCubicCoefficients(yCoeffsH, yIdxDiffH);

        float yCoeffsW[4];
        CalcCubicCoefficients(yCoeffsW, yIdxDiffW);

        float gradsValue = static_cast<float>(grads[gradsIdx]);
        for (T_IDX2 i = 0; i < 4; i++) {
            for (T_IDX2 j = 0; j < 4; j++) {
                CalcOutputValue<T_DATA, T_IDX, T_IDX2, FORMAT>(y,
                    lenC,
                    lenSrcH,
                    lenSrcW,
                    idxN,
                    idxC,
                    yFloorIdxH - 1 + i,
                    yFloorIdxW - 1 + j,
                    gradsValue * yCoeffsH[i] * yCoeffsW[j]);
            }
        }
    }
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT32) __aicore__ void calleeSimtInt32(__gm__ T_DATA *grads, __gm__ T_DATA *y,
    T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtCompute<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
        y,
        lenC,
        lenSrcH,
        lenSrcW,
        lenDstH,
        lenDstW,
        scaleH,
        scaleW,
        coreFactor,
        coreOffset,
        mC,
        shiftC,
        mH,
        shiftH,
        mW,
        shiftW);
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT64) __aicore__ void calleeSimtInt64(__gm__ T_DATA *grads, __gm__ T_DATA *y,
    T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW,
    T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtCompute<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
        y,
        lenC,
        lenSrcH,
        lenSrcW,
        lenDstH,
        lenDstW,
        scaleH,
        scaleW,
        coreFactor,
        coreOffset,
        mC,
        shiftC,
        mH,
        shiftH,
        mW,
        shiftW);
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ inline void ResizeBicubicV2GradSimt<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>::Process()
{
    const T_IDX lenC = tilingData_->lenC;
    const T_IDX lenSrcW = tilingData_->lenSrcW;
    const T_IDX lenSrcH = tilingData_->lenSrcH;
    const T_IDX lenDstW = tilingData_->lenDstW;
    const T_IDX lenDstH = tilingData_->lenDstH;
    const int32_t initYUseCoreNum = tilingData_->initYUseCoreNum;
    const int32_t useCoreNum = tilingData_->useCoreNum;
    const float scaleH = tilingData_->scaleH;
    const float scaleW = tilingData_->scaleW;

    T_IDX initYCoreFactor = tilingData_->initYCoreFactor;
    T_IDX initYCoreOffset = coreIdx_ * tilingData_->initYCoreFactor;
    if (coreIdx_ < tilingData_->initYCoreTailFactor) {
        initYCoreFactor += 1;
        initYCoreOffset += coreIdx_;
    } else {
        initYCoreOffset += tilingData_->initYCoreTailFactor;
    }

    T_IDX coreFactor = tilingData_->coreFactor;
    T_IDX coreOffset = coreIdx_ * tilingData_->coreFactor;
    if (coreIdx_ < tilingData_->coreTailFactor) {
        coreFactor += 1;
        coreOffset += coreIdx_;
    } else {
        coreOffset += tilingData_->coreTailFactor;
    }

    T_IDX mC = 1, mH = 1, mW = 1;
    T_IDX shiftC = 1, shiftH = 1, shiftW = 1;
    GetUintDivMagicAndShift(mC, shiftC, lenC);
    GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);

    if (coreIdx_ < initYUseCoreNum) {
        InitOutput<T_DATA>(yGm_[initYCoreOffset], initYCoreFactor, static_cast<T_DATA>(0.0f));
    }
    SyncAll();

    if (coreIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtInt32<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT32),
                (__gm__ T_DATA *)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA *)(yGm_.GetPhyAddr()),
                lenC,
                lenSrcH,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                coreFactor,
                coreOffset,
                mC,
                shiftC,
                mH,
                shiftH,
                mW,
                shiftW);
        } else {
            Simt::VF_CALL<calleeSimtInt64<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT64),
                (__gm__ T_DATA *)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA *)(yGm_.GetPhyAddr()),
                lenC,
                lenSrcH,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                coreFactor,
                coreOffset,
                mC,
                shiftC,
                mH,
                shiftH,
                mW,
                shiftW);
        }
    }
}
}  // namespace ResizeBicubicV2Grad

#endif  // CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_H
