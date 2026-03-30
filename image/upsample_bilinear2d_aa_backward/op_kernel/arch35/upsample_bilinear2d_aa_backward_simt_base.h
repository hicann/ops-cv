/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bilinear2d_aa_backward_simt_base.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_BASE_H
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bilinear2d_aa_backward_tiling_data.h"

namespace UpsampleBilinear2dAABackward {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 512;
const int32_t THREAD_NUM_B64 = 512;

template <typename T2, typename T3>
static __simt_callee__ __aicore__ inline T3 GetLeftIndex(T2 index, float scale, float support)
{
    T3 leftIndex = 0;
    if (scale != 0.0f) {
        if (index > 0) {
            index -= 1;
        }
        leftIndex = Simt::Max(static_cast<T3>(static_cast<float>(index) / scale - support) - 1, static_cast<T3>(0));
    }
    return leftIndex;
}

static __simt_callee__ __aicore__ inline float WeightCalculate(float x)
{
    x = Simt::Abs(x);
    if (x < 1.0f) {
        return 1.0f - x;
    }
    return 0.0f;
}

template <typename T2, typename T3>
static __simt_callee__ __aicore__ inline float GetWeights(T2 index, T3 min, T3 max, float center, float invScale) {
    float totalWeights = 0.0f;
    for (T3 j = min; j < max; j++) {
        const float distance = (static_cast<float>(j) - center + 0.5f) * invScale;
        const float w = WeightCalculate(distance);
        totalWeights += w;
    }
    const float distance = (static_cast<float>(index) - center + 0.5f) * invScale;
    float weight = WeightCalculate(distance);
    if (totalWeights != 0.0f) {
        weight /= totalWeights;
    }
    return weight;
}

template <typename T1, typename T2, typename T3>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T3 lenSrcH, T3 lenSrcW, T3 lenDstH, T3 lenDstW, T3 maxInterpSizeH, T3 maxInterpSizeW,
    float scaleH, float scaleW, float invScaleH, float invScaleW, float supportH, float supportW)
{
    T3 lenSrcHw = lenSrcH * lenSrcW;
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        T3 yGmIdx = blkStartOffset + idx;
        float inputValue = static_cast<float>(inputGm[yGmIdx]);
        T2 tmpRes = Simt::UintDiv(static_cast<T2>(yGmIdx), mW, shiftW);
        T2 W = yGmIdx - tmpRes * lenDstW;
        T2 NC = Simt::UintDiv(tmpRes, mH, shiftH);
        T2 H = tmpRes - NC * lenDstH;

        float centerH = scaleH * (static_cast<float>(H) + 0.5f);
        T3 minH = Simt::Max(static_cast<T3>(Simt::Floor(centerH - supportH + 0.5f)), static_cast<T3>(0));
        T3 maxH = Simt::Min(static_cast<T3>(Simt::Floor(centerH + supportH + 0.5f)), lenSrcH);
        float totalWeightsH = 0.0f;
        for (T3 j = minH; j < maxH; j++) {
            const float distance = (static_cast<float>(j) - centerH + 0.5f) * invScaleH;
            const float w = WeightCalculate(distance);
            totalWeightsH += w;
        }

        float centerW = scaleW * (static_cast<float>(W) + 0.5f);
        T3 minW = Simt::Max(static_cast<T3>(Simt::Floor(centerW - supportW + 0.5f)), static_cast<T3>(0));
        T3 maxW = Simt::Min(static_cast<T3>(Simt::Floor(centerW + supportW + 0.5f)), lenSrcW);
        float totalWeightsW = 0.0f;
        for (T3 i = minW; i < maxW; i++) {
            const float distance = (static_cast<float>(i) - centerW + 0.5f) * invScaleW;
            const float w = WeightCalculate(distance);
            totalWeightsW += w;
        }

        for (T3 h = minH; h < maxH; h++) {
            const float distanceH = (static_cast<float>(h) - centerH + 0.5f) * invScaleH;
            float weightH = WeightCalculate(distanceH);
            if (totalWeightsH != 0.0f) {
                weightH /= totalWeightsH;
            }
            for (T3 w = minW; w < maxW; w++) {
                const float distanceW = (static_cast<float>(w) - centerW + 0.5f) * invScaleW;
                float weightW = WeightCalculate(distanceW);
                if (totalWeightsW != 0.0f) {
                    weightW /= totalWeightsW;
                }
                const T3 input_idx = NC * lenSrcHw + h * lenSrcW + w;
                Simt::AtomicAdd(outputGm + input_idx, static_cast<T1>(inputValue * weightH * weightW));
            }
        }
    }
}

template <typename T1, typename T2, typename T3>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtComputeDetermine(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T3 lenSrcH, T3 lenSrcW, T3 lenDstH, T3 lenDstW, T3 maxInterpSizeH, T3 maxInterpSizeW,
    float scaleH, float scaleW, float invScaleH, float invScaleW, float supportH, float supportW)
{
    T3 lenSrcHw = lenSrcH * lenSrcW;
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        T3 yGmIdx = blkStartOffset + idx;
        T2 tmpRes = Simt::UintDiv(static_cast<T2>(yGmIdx), mW, shiftW);
        T2 W = yGmIdx - tmpRes * lenDstW;
        T2 NC = Simt::UintDiv(tmpRes, mH, shiftH);
        T2 H = tmpRes - NC * lenDstH;

        T3 leftH = GetLeftIndex<T2, T3>(H, scaleH, supportH);
        T3 leftW = GetLeftIndex<T2, T3>(W, scaleW, supportW);
        T3 rightH = Simt::Min(leftH + maxInterpSizeH, lenSrcH);
        T3 rightW = Simt::Min(leftW + maxInterpSizeW, lenSrcW);
        float value = 0.0f;
        for (T3 h = leftH; h < rightH; h++) {
            float centerH = scaleH * (static_cast<float>(h) + 0.5f);
            T3 minH = Simt::Max(static_cast<T3>(Simt::Floor(centerH - supportH + 0.5f)), static_cast<T3>(0));
            T3 maxH = Simt::Min(static_cast<T3>(Simt::Floor(centerH + supportH + 0.5f)), lenDstH);
            if (H >= maxH) {
                continue;
            }
            if (H < minH) {
                break;
            }
            float weightH = GetWeights<T2, T3>(H, minH, maxH, centerH, invScaleH);
            for (T3 w = leftW; w < rightW; w++) {
                float centerW = scaleW * (static_cast<float>(w) + 0.5f);
                T3 minW = Simt::Max(static_cast<T3>(Simt::Floor(centerW - supportW + 0.5f)), static_cast<T3>(0));
                T3 maxW = Simt::Min(static_cast<T3>(Simt::Floor(centerW + supportW + 0.5f)), lenDstW);
                if (W >= maxW) {
                    continue;
                }
                if (W < minW) {
                    break;
                }
                float weightW = GetWeights<T2, T3>(W, minW, maxW, centerW, invScaleW);
                const T3 input_idx = NC * lenSrcHw + h * lenSrcW + w;
                value += inputGm[input_idx] * weightH * weightW;
            }
        }
        outputGm[yGmIdx] = value;
    }
}

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__ void calleeInt32(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T3 lenSrcH, T3 lenSrcW, T3 lenDstH, T3 lenDstW, T3 maxInterpSizeH, T3 maxInterpSizeW,
    float scaleH, float scaleW, float invScaleH, float invScaleW, float supportH, float supportW)
{
    if constexpr (isDetermine == 0) {
        SimtCompute<T1, T2, T3>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, 
            lenSrcH, lenSrcW, lenDstH, lenDstW, maxInterpSizeH, maxInterpSizeW, scaleH, scaleW, invScaleH, invScaleW, 
            supportH, supportW);
    } else {
        SimtComputeDetermine<T1, T2, T3>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, 
            mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, maxInterpSizeH, maxInterpSizeW, scaleH, scaleW, 
            invScaleH, invScaleW, supportH, supportW);
    }
}

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__ void calleeInt64(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T3 lenSrcH, T3 lenSrcW, T3 lenDstH, T3 lenDstW, T3 maxInterpSizeH, T3 maxInterpSizeW,
    float scaleH, float scaleW, float invScaleH, float invScaleW, float supportH, float supportW)
{
    if constexpr (isDetermine == 0) {
        SimtCompute<T1, T2, T3>(
            inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH,
            lenDstW, maxInterpSizeH, maxInterpSizeW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
    } else {
        SimtComputeDetermine<T1, T2, T3>(
            inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH,
            lenDstW, maxInterpSizeH, maxInterpSizeW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
    }
}
} // namespace UpsampleBilinear2dAABackward

#endif // UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_BASE_H
