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
 * \file resize_bicubic_v2_grad_simt_determine.h
 * \brief resize_bicubic_v2_grad_simt_determine
 */

#ifndef CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_DETERMINE_H
#define CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_DETERMINE_H

#include "resize_bicubic_v2_grad_base.h"

namespace ResizeBicubicV2Grad {
using namespace AscendC;

constexpr int32_t SIMT_DETERMINE_THREAD_NUM_INT32 = 512;
constexpr int32_t SIMT_DETERMINE_THREAD_NUM_INT64 = 256;

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
class ResizeBicubicV2GradSimtDetermine {
public:
    __aicore__ inline ResizeBicubicV2GradSimtDetermine(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR y, const ResizeBicubicV2GradSimtDetermineTilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T_DATA> gradsGm_;
    GlobalTensor<T_DATA> yGm_;
    int32_t coreIdx_;
    const ResizeBicubicV2GradSimtDetermineTilingData *tilingData_;
};

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ inline void ResizeBicubicV2GradSimtDetermine<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>::Init(
    GM_ADDR grads, GM_ADDR y, const ResizeBicubicV2GradSimtDetermineTilingData *tilingData)
{
    coreIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    gradsGm_.SetGlobalBuffer((__gm__ T_DATA *)grads);
    yGm_.SetGlobalBuffer((__gm__ T_DATA *)y);
}

template <typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline void CalcOutputDimIdx(T_IDX yIdx, T_IDX lenC, T_IDX lenSrcH,
    T_IDX lenSrcW, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW, T_IDX &idxN, T_IDX &idxC,
    T_IDX &yIdxH, T_IDX &yIdxW)
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

template <typename T_IDX, typename T_IDX2, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline void CalcIdxStart(float scaleH, float scaleW, float inverseScaleH,
    float inverseScaleW, T_IDX yIdxH, T_IDX yIdxW, T_IDX &gradsIdxHStart, float &yRealIdxHStart, T_IDX &gradsIdxWStart,
    float &yRealIdxWStart)
{
    float offset = 0.5f;
    if constexpr (ALIGN_CORNERS) {
        offset = 0.0f;
    }

    gradsIdxHStart = static_cast<T_IDX>(Simt::Max(static_cast<T_IDX2>(0),
        static_cast<T_IDX2>(Simt::Ceil((static_cast<float>(yIdxH) - 2.0f + offset) * inverseScaleH - offset))));
    yRealIdxHStart = (static_cast<float>(gradsIdxHStart) + offset) * scaleH - offset;
    gradsIdxWStart = static_cast<T_IDX>(Simt::Max(static_cast<T_IDX2>(0),
        static_cast<T_IDX2>(Simt::Ceil((static_cast<float>(yIdxW) - 2.0f + offset) * inverseScaleW - offset))));
    yRealIdxWStart = (static_cast<float>(gradsIdxWStart) + offset) * scaleW - offset;
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT>
__aicore__ __attribute__((always_inline)) inline float GetInputValue(__gm__ T_DATA *grads, T_IDX lenC, T_IDX lenDstH,
    T_IDX lenDstW, T_IDX idxN, T_IDX idxC, T_IDX gradsIdxH, T_IDX gradsIdxW)
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

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline float CalcInternalPoint(__gm__ T_DATA *grads, T_IDX lenC,
    T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX idxN, T_IDX idxC, T_IDX yIdxH, T_IDX yIdxW,
    T_IDX gradsIdxHStart, float yRealIdxHStart, T_IDX gradsIdxWStart, float yRealIdxWStart)
{
    float addValue = 0.0f;

    float yRealIdxH = yRealIdxHStart;
    T_IDX gradsIdxH = gradsIdxHStart;
    while (yRealIdxH < (static_cast<float>(yIdxH) + 2.0f) && gradsIdxH < lenDstH) {
        float yIdxDiffH = Simt::Abs(yRealIdxH - static_cast<float>(yIdxH));
        float yCoeffH = CalcCubicCoefficient(yIdxDiffH);

        float yRealIdxW = yRealIdxWStart;
        T_IDX gradsIdxW = gradsIdxWStart;
        while (yRealIdxW < (static_cast<float>(yIdxW) + 2.0f) && gradsIdxW < lenDstW) {
            float yIdxDiffW = Simt::Abs(yRealIdxW - static_cast<float>(yIdxW));
            float yCoeffW = CalcCubicCoefficient(yIdxDiffW);

            float gradsValue =
                GetInputValue<T_DATA, T_IDX, FORMAT>(grads, lenC, lenDstH, lenDstW, idxN, idxC, gradsIdxH, gradsIdxW);

            addValue += gradsValue * yCoeffH * yCoeffW;

            gradsIdxW++;
            yRealIdxW = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleW, gradsIdxW);
        }
        gradsIdxH++;
        yRealIdxH = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleH, gradsIdxH);
    }

    return addValue;
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline float CalcWBoundaryPoint(__gm__ T_DATA *grads, T_IDX lenC,
    T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX idxN, T_IDX idxC, T_IDX yIdxH,
    T_IDX yIdxW, T_IDX gradsIdxHStart, float yRealIdxHStart, T_IDX gradsIdxWStart, float yRealIdxWStart)
{
    float addValue = 0.0f;

    float yRealIdxH = yRealIdxHStart;
    T_IDX gradsIdxH = gradsIdxHStart;
    while (yRealIdxH < (static_cast<float>(yIdxH) + 2.0f) && gradsIdxH < lenDstH) {
        float yIdxDiffH = Simt::Abs(yRealIdxH - static_cast<float>(yIdxH));
        float yCoeffH = CalcCubicCoefficient(yIdxDiffH);

        float yRealIdxW = yRealIdxWStart;
        T_IDX gradsIdxW = gradsIdxWStart;
        while (gradsIdxW < lenDstW) {
            T_IDX2 yFloorIdxW = static_cast<T_IDX2>(Simt::Floor(yRealIdxW));
            if (yIdxW == 0 && yFloorIdxW > 1 && lenSrcW > 1) {
                break;
            }
            float yIdxDiffW = yRealIdxW - static_cast<float>(yFloorIdxW);
            float yCoeffsW[4];
            CalcCubicCoefficients(yCoeffsW, yIdxDiffW);

            float gradsValue =
                GetInputValue<T_DATA, T_IDX, FORMAT>(grads, lenC, lenDstH, lenDstW, idxN, idxC, gradsIdxH, gradsIdxW);

            for (T_IDX2 j = 0; j < 4; j++) {
                T_IDX yTempIdxW = static_cast<T_IDX>(
                    Simt::Max(static_cast<T_IDX2>(0), Simt::Min(yFloorIdxW - 1 + j, static_cast<T_IDX2>(lenSrcW) - 1)));
                if (yTempIdxW == yIdxW) {
                    addValue += gradsValue * yCoeffH * yCoeffsW[j];
                }
            }

            gradsIdxW++;
            yRealIdxW = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleW, gradsIdxW);
        }
        gradsIdxH++;
        yRealIdxH = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleH, gradsIdxH);
    }

    return addValue;
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline float CalcHBoundaryPoint(__gm__ T_DATA *grads, T_IDX lenC,
    T_IDX lenSrcH, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX idxN, T_IDX idxC, T_IDX yIdxH,
    T_IDX yIdxW, T_IDX gradsIdxHStart, float yRealIdxHStart, T_IDX gradsIdxWStart, float yRealIdxWStart)
{
    float addValue = 0.0f;

    float yRealIdxH = yRealIdxHStart;
    T_IDX gradsIdxH = gradsIdxHStart;
    while (gradsIdxH < lenDstH) {
        T_IDX2 yFloorIdxH = static_cast<T_IDX2>(Simt::Floor(yRealIdxH));
        if (yIdxH == 0 && yFloorIdxH > 1 && lenSrcH > 1) {
            break;
        }
        float yIdxDiffH = yRealIdxH - static_cast<float>(yFloorIdxH);
        float yCoeffsH[4];
        CalcCubicCoefficients(yCoeffsH, yIdxDiffH);

        float yRealIdxW = yRealIdxWStart;
        T_IDX gradsIdxW = gradsIdxWStart;
        while (yRealIdxW < (static_cast<float>(yIdxW) + 2.0f) && gradsIdxW < lenDstW) {
            float yIdxDiffW = Simt::Abs(yRealIdxW - static_cast<float>(yIdxW));
            float yCoeffW = CalcCubicCoefficient(yIdxDiffW);

            float gradsValue =
                GetInputValue<T_DATA, T_IDX, FORMAT>(grads, lenC, lenDstH, lenDstW, idxN, idxC, gradsIdxH, gradsIdxW);

            for (T_IDX2 i = 0; i < 4; i++) {
                T_IDX yTempIdxH = static_cast<T_IDX>(
                    Simt::Max(static_cast<T_IDX2>(0), Simt::Min(yFloorIdxH - 1 + i, static_cast<T_IDX2>(lenSrcH) - 1)));
                if (yTempIdxH == yIdxH) {
                    addValue += gradsValue * yCoeffsH[i] * yCoeffW;
                }
            }

            gradsIdxW++;
            yRealIdxW = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleW, gradsIdxW);
        }
        gradsIdxH++;
        yRealIdxH = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleH, gradsIdxH);
    }

    return addValue;
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline float CalcCornerPoint(__gm__ T_DATA *grads, T_IDX lenC, T_IDX lenSrcH,
    T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX idxN, T_IDX idxC, T_IDX yIdxH,
    T_IDX yIdxW, T_IDX gradsIdxHStart, float yRealIdxHStart, T_IDX gradsIdxWStart, float yRealIdxWStart)
{
    float addValue = 0.0f;

    float yRealIdxH = yRealIdxHStart;
    T_IDX gradsIdxH = gradsIdxHStart;
    while (gradsIdxH < lenDstH) {
        T_IDX2 yFloorIdxH = static_cast<T_IDX2>(Simt::Floor(yRealIdxH));
        if (yIdxH == 0 && yFloorIdxH > 1 && lenSrcH > 1) {
            break;
        }
        float yIdxDiffH = yRealIdxH - static_cast<float>(yFloorIdxH);
        float yCoeffsH[4];
        CalcCubicCoefficients(yCoeffsH, yIdxDiffH);

        float yRealIdxW = yRealIdxWStart;
        T_IDX gradsIdxW = gradsIdxWStart;
        while (gradsIdxW < lenDstW) {
            T_IDX2 yFloorIdxW = static_cast<T_IDX2>(Simt::Floor(yRealIdxW));
            if (yIdxW == 0 && yFloorIdxW > 1 && lenSrcW > 1) {
                break;
            }
            float yIdxDiffW = yRealIdxW - static_cast<float>(yFloorIdxW);
            float yCoeffsW[4];
            CalcCubicCoefficients(yCoeffsW, yIdxDiffW);

            float gradsValue =
                GetInputValue<T_DATA, T_IDX, FORMAT>(grads, lenC, lenDstH, lenDstW, idxN, idxC, gradsIdxH, gradsIdxW);

            for (T_IDX2 i = 0; i < 4; i++) {
                T_IDX yTempIdxH = static_cast<T_IDX>(
                    Simt::Max(static_cast<T_IDX2>(0), Simt::Min(yFloorIdxH - 1 + i, static_cast<T_IDX2>(lenSrcH) - 1)));
                for (T_IDX2 j = 0; j < 4; j++) {
                    T_IDX yTempIdxW = static_cast<T_IDX>(Simt::Max(
                        static_cast<T_IDX2>(0), Simt::Min(yFloorIdxW - 1 + j, static_cast<T_IDX2>(lenSrcW) - 1)));
                    if (yTempIdxH == yIdxH && yTempIdxW == yIdxW) {
                        addValue += gradsValue * yCoeffsH[i] * yCoeffsW[j];
                    }
                }
            }

            gradsIdxW++;
            yRealIdxW = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleW, gradsIdxW);
        }
        gradsIdxH++;
        yRealIdxH = CalcSourceIndex<T_IDX, ALIGN_CORNERS>(scaleH, gradsIdxH);
    }

    return addValue;
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__aicore__ __attribute__((always_inline)) inline void SimtDetermineCompute(__gm__ T_DATA *grads, __gm__ T_DATA *y,
    T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW,
    float inverseScaleH, float inverseScaleW, T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH,
    T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yIdx = coreOffset + idx;
        T_IDX idxN = 0, idxC = 0, yIdxH = 0, yIdxW = 0;
        CalcOutputDimIdx<T_IDX, FORMAT>(
            yIdx, lenC, lenSrcH, lenSrcW, mC, shiftC, mH, shiftH, mW, shiftW, idxN, idxC, yIdxH, yIdxW);

        T_IDX gradsIdxHStart = 0, gradsIdxWStart = 0;
        float yRealIdxHStart = 0.0f, yRealIdxWStart = 0.0f;
        CalcIdxStart<T_IDX, T_IDX2, ALIGN_CORNERS>(scaleH,
            scaleW,
            inverseScaleH,
            inverseScaleW,
            yIdxH,
            yIdxW,
            gradsIdxHStart,
            yRealIdxHStart,
            gradsIdxWStart,
            yRealIdxWStart);

        float addValue = 0.0f;
        if ((yIdxH != 0 && yIdxH != lenSrcH - 1) && (yIdxW != 0 && yIdxW != lenSrcW - 1)) {
            addValue = CalcInternalPoint<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS>(grads,
                lenC,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                idxN,
                idxC,
                yIdxH,
                yIdxW,
                gradsIdxHStart,
                yRealIdxHStart,
                gradsIdxWStart,
                yRealIdxWStart);
        } else if ((yIdxH != 0 && yIdxH != lenSrcH - 1) && (yIdxW == 0 || yIdxW == lenSrcW - 1)) {
            addValue = CalcWBoundaryPoint<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
                lenC,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                idxN,
                idxC,
                yIdxH,
                yIdxW,
                gradsIdxHStart,
                yRealIdxHStart,
                gradsIdxWStart,
                yRealIdxWStart);
        } else if ((yIdxH == 0 || yIdxH == lenSrcH - 1) && (yIdxW != 0 && yIdxW != lenSrcW - 1)) {
            addValue = CalcHBoundaryPoint<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
                lenC,
                lenSrcH,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                idxN,
                idxC,
                yIdxH,
                yIdxW,
                gradsIdxHStart,
                yRealIdxHStart,
                gradsIdxWStart,
                yRealIdxWStart);
        } else {
            addValue = CalcCornerPoint<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
                lenC,
                lenSrcH,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                idxN,
                idxC,
                yIdxH,
                yIdxW,
                gradsIdxHStart,
                yRealIdxHStart,
                gradsIdxWStart,
                yRealIdxWStart);
        }

        y[yIdx] = static_cast<T_DATA>(addValue);
    }
}

template <typename T_DATA, typename T_IDX, typename T_IDX2, int32_t FORMAT, bool ALIGN_CORNERS>
__simt_vf__ LAUNCH_BOUND(SIMT_DETERMINE_THREAD_NUM_INT32) __aicore__
    void calleeSimtDetermineInt32(__gm__ T_DATA *grads, __gm__ T_DATA *y, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW,
        T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, float inverseScaleH, float inverseScaleW,
        T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineCompute<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
        y,
        lenC,
        lenSrcH,
        lenSrcW,
        lenDstH,
        lenDstW,
        scaleH,
        scaleW,
        inverseScaleH,
        inverseScaleW,
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
__simt_vf__ LAUNCH_BOUND(SIMT_DETERMINE_THREAD_NUM_INT64) __aicore__
    void calleeSimtDetermineInt64(__gm__ T_DATA *grads, __gm__ T_DATA *y, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW,
        T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, float inverseScaleH, float inverseScaleW,
        T_IDX coreFactor, T_IDX coreOffset, T_IDX mC, T_IDX shiftC, T_IDX mH, T_IDX shiftH, T_IDX mW, T_IDX shiftW)
{
    SimtDetermineCompute<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>(grads,
        y,
        lenC,
        lenSrcH,
        lenSrcW,
        lenDstH,
        lenDstW,
        scaleH,
        scaleW,
        inverseScaleH,
        inverseScaleW,
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
__aicore__ inline void ResizeBicubicV2GradSimtDetermine<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>::Process()
{
    const T_IDX lenC = tilingData_->lenC;
    const T_IDX lenSrcH = tilingData_->lenSrcH;
    const T_IDX lenSrcW = tilingData_->lenSrcW;
    const T_IDX lenDstH = tilingData_->lenDstH;
    const T_IDX lenDstW = tilingData_->lenDstW;
    const int32_t useCoreNum = tilingData_->useCoreNum;
    const float scaleH = tilingData_->scaleH;
    const float scaleW = tilingData_->scaleW;
    const float inverseScaleH = tilingData_->inverseScaleH;
    const float inverseScaleW = tilingData_->inverseScaleW;

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
    GetUintDivMagicAndShift(mH, shiftH, lenSrcH);
    GetUintDivMagicAndShift(mW, shiftW, lenSrcW);

    if (coreIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtDetermineInt32<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>>(
                Simt::Dim3(SIMT_DETERMINE_THREAD_NUM_INT32),
                (__gm__ T_DATA *)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA *)(yGm_.GetPhyAddr()),
                lenC,
                lenSrcH,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                inverseScaleH,
                inverseScaleW,
                coreFactor,
                coreOffset,
                mC,
                shiftC,
                mH,
                shiftH,
                mW,
                shiftW);
        } else {
            Simt::VF_CALL<calleeSimtDetermineInt64<T_DATA, T_IDX, T_IDX2, FORMAT, ALIGN_CORNERS>>(
                Simt::Dim3(SIMT_DETERMINE_THREAD_NUM_INT64),
                (__gm__ T_DATA *)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA *)(yGm_.GetPhyAddr()),
                lenC,
                lenSrcH,
                lenSrcW,
                lenDstH,
                lenDstW,
                scaleH,
                scaleW,
                inverseScaleH,
                inverseScaleW,
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

#endif  // CANN_RESIZE_BICUBIC_V2_GRAD_SIMT_DETERMINE_H