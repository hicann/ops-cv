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
 * \file upsample_nearest_exact2d_grad_simt_base.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT_BASE_H
#define UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest_exact2d_grad_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

namespace UpsampleNearestExact2dGrad {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 2048;
const int32_t THREAD_NUM_B64 = 1024;
const uint64_t SCH_ID_1 = 1;
const uint64_t SCH_ID_2 = 2;
const uint64_t SCH_ID_3 = 3;

template <typename T2, bool isExact>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeOrig(T2 idx, T2 limit, float scale,
                                                                                  T2& orig)
{
    if constexpr (isExact) {
        orig = ceilf((static_cast<float>(idx) * scale - 0.5f));
    } else {
        orig = ceilf((static_cast<float>(idx) * scale));
    }
    orig = min(orig, limit);
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeForSplitNchw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenSrcW, T2 lenSrcHw, T2 N, T2 origH, T2 origHUp, T2 origW, T2 origWUp,
    T2 yGmIdx)
{
    float grad = 0;
    for (T2 h = origH; h < origHUp; h++) {
        for (T2 w = origW; w < origWUp; w++) {
            T2 srcOffset = N * lenSrcHw + h * lenSrcW + w;
            grad += static_cast<float>(inputGm[srcOffset]);
        }
    }
    outputGm[yGmIdx] = static_cast<T1>(grad);
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeForSplitHw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenN, T2 lenSrcW, T2 lenSrcHw, T2 lenDstHw, T2 origH, T2 origHUp,
    T2 origW, T2 origWUp, T2 yGmIdx)
{
    for (T2 nc = 0; nc < lenN; nc++) {
        float grad = 0;
        T2 srcNcOffset = nc * lenSrcHw;
        for (T2 h = origH; h < origHUp; h++) {
            for (T2 w = origW; w < origWUp; w++) {
                T2 srcOffset = srcNcOffset + h * lenSrcW + w;
                grad += static_cast<float>(inputGm[srcOffset]);
            }
        }
        T2 outOffset = nc * lenDstHw + yGmIdx;
        outputGm[outOffset] = static_cast<T1>(grad);
    }
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeForSplitChw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenN, T2 lenC, T2 lenSrcW, T2 lenSrcHw, T2 lenDstHw, T2 C, T2 origH,
    T2 origHUp, T2 origW, T2 origWUp, T2 yGmIdx)
{
    for (T2 n = 0; n < lenN; n++) {
        float grad = 0;
        T2 srcNcOffset = (n * lenC + C) * lenSrcHw;
        for (T2 h = origH; h < origHUp; h++) {
            for (T2 w = origW; w < origWUp; w++) {
                T2 srcOffset = srcNcOffset + h * lenSrcW + w;
                grad += static_cast<float>(inputGm[srcOffset]);
            }
        }
        T2 outOffset = n * lenC * lenDstHw + yGmIdx;
        outputGm[outOffset] = static_cast<T1>(grad);
    }
}

template <typename T1, typename T2, bool isExact, uint64_t schId>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcH, T2 lenSrcW, T2 lenDstH, T2 lenDstW, float scaleH, float scaleW)
{
    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += static_cast<T2>(blockDim.x)) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 W = 0, H = 0, C = 0, N = 0;
        T2 tempRes = Simt::UintDiv(yGmIdx, mW, shiftW);
        W = yGmIdx - tempRes * lenDstW;
        if constexpr (schId == SCH_ID_1) {
            H = tempRes;
        }
        if constexpr (schId == SCH_ID_2) {
            C = Simt::UintDiv(tempRes, mH, shiftH);
            H = tempRes - C * lenDstH;
        }
        if constexpr (schId == SCH_ID_3) {
            N = Simt::UintDiv(tempRes, mH, shiftH);
            H = tempRes - N * lenDstH;
        }
        T2 origH = 0, origW = 0, origHUp = 0, origWUp = 0;
        ComputeOrig<T2, isExact>(H, lenSrcH, scaleH, origH);
        ComputeOrig<T2, isExact>(W, lenSrcW, scaleW, origW);
        ComputeOrig<T2, isExact>(H + 1, lenSrcH, scaleH, origHUp);
        ComputeOrig<T2, isExact>(W + 1, lenSrcW, scaleW, origWUp);
        T2 lenSrcHw = lenSrcH * lenSrcW;
        T2 lenDstHw = lenDstH * lenDstW;
        if constexpr (schId == SCH_ID_3) {
            ComputeForSplitNchw<T1, T2>(inputGm, outputGm, lenSrcW, lenSrcHw, N, origH, origHUp, origW, origWUp,
                                        yGmIdx);
        }
        if constexpr (schId == SCH_ID_1) {
            ComputeForSplitHw<T1, T2>(inputGm, outputGm, lenN, lenSrcW, lenSrcHw, lenDstHw, origH, origHUp, origW,
                                      origWUp, yGmIdx);
        }
        if constexpr (schId == SCH_ID_2) {
            ComputeForSplitChw<T1, T2>(inputGm, outputGm, lenN, lenC, lenSrcW, lenSrcHw, lenDstHw, C, origH, origHUp,
                                       origW, origWUp, yGmIdx);
        }
    }
}

template <typename T1, typename T2, bool isExact, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__
    void calleeInt32(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcH, T2 lenSrcW, T2 lenDstH, T2 lenDstW, float scaleH,
                     float scaleW)
{
    SimtCompute<T1, T2, isExact, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW,
                                        shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW);
}

template <typename T1, typename T2, bool isExact, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__
    void calleeInt64(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcH, T2 lenSrcW, T2 lenDstH, T2 lenDstW, float scaleH,
                     float scaleW)
{
    SimtCompute<T1, T2, isExact, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW,
                                        shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW);
}
} // namespace UpsampleNearestExact2dGrad
#endif // UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT_BASE_H
