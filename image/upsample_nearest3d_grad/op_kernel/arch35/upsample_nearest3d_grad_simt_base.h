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
 * \file upsample_nearest3d_grad_simt_base.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_GRAD_SIMT_BASE_H
#define UPSAMPLE_NEAREST3D_GRAD_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest3d_grad_tiling_data.h"

namespace UpsampleNearest3dGrad {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 2048;
const int32_t THREAD_NUM_B64 = 1024;
const uint64_t SCH_ID_1 = 1;
const uint64_t SCH_ID_2 = 2;
const uint64_t SCH_ID_3 = 3;

template <typename T2, bool isExtra>
__aicore__ __attribute__((always_inline)) inline void ComputeOrig(T2 idx, T2 limit, float scale, T2& orig)
{
    if constexpr (isExtra) {
        orig = Simt::Ceil((static_cast<float>(idx) * scale - 0.5f));
    } else {
        orig = Simt::Ceil((static_cast<float>(idx) * scale));
    }
    orig = Simt::Min(orig, limit);
}

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void ComputeForSplitNcdhw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenSrcW, T2 lenSrcHw, T2 lenSrcDhw, T2 N, T2 origD, T2 origDUp,
    T2 origH, T2 origHUp, T2 origW, T2 origWUp, T2 yGmIdx)
{
    float grad = 0;
    for (T2 d = origD; d < origDUp; d++) {
        for (T2 h = origH; h < origHUp; h++) {
            for (T2 w = origW; w < origWUp; w++) {
                T2 srcOffset = N * lenSrcDhw + d * lenSrcHw + h * lenSrcW + w;
                grad += static_cast<float>(inputGm[srcOffset]);
            }
        }
    }
    outputGm[yGmIdx] = static_cast<T1>(grad);
}

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void ComputeForSplitDhw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenN, T2 lenSrcW, T2 lenSrcHw, T2 lenSrcDhw, T2 lenDstDhw, T2 origD,
    T2 origDUp, T2 origH, T2 origHUp, T2 origW, T2 origWUp, T2 yGmIdx)
{
    for (T2 nc = 0; nc < lenN; nc++) {
        float grad = 0;
        T2 srcNcOffset = nc * lenSrcDhw;
        for (T2 d = origD; d < origDUp; d++) {
            for (T2 h = origH; h < origHUp; h++) {
                for (T2 w = origW; w < origWUp; w++) {
                    T2 srcOffset = srcNcOffset + d * lenSrcHw + h * lenSrcW + w;
                    grad += static_cast<float>(inputGm[srcOffset]);
                }
            }
        }
        T2 outOffset = nc * lenDstDhw + yGmIdx;
        outputGm[outOffset] = static_cast<T1>(grad);
    }
}

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void ComputeForSplitCdhw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 lenN, T2 lenC, T2 lenSrcW, T2 lenSrcHw, T2 lenSrcDhw, T2 lenDstDhw,
    T2 C, T2 origD, T2 origDUp, T2 origH, T2 origHUp, T2 origW, T2 origWUp, T2 yGmIdx)
{
    for (T2 n = 0; n < lenN; n++) {
        float grad = 0;
        T2 srcNcOffset = (n * lenC + C) * lenSrcDhw;
        for (T2 d = origD; d < origDUp; d++) {
            for (T2 h = origH; h < origHUp; h++) {
                for (T2 w = origW; w < origWUp; w++) {
                    T2 srcOffset = srcNcOffset + d * lenSrcHw + h * lenSrcW + w;
                    grad += static_cast<float>(inputGm[srcOffset]);
                }
            }
        }
        T2 outOffset = n * lenC * lenDstDhw + yGmIdx;
        outputGm[outOffset] = static_cast<T1>(grad);
    }
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD,
    T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW)
{
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 W = 0, H = 0, D = 0, C = 0, N = 0;
        T2 tempRes = Simt::UintDiv(yGmIdx, mW, shiftW);
        W = yGmIdx - tempRes * lenDstW;
        if constexpr (schId == SCH_ID_1) {
            D = Simt::UintDiv(tempRes, mH, shiftH);
            H = tempRes - D * lenDstH;
        }
        if constexpr (schId == SCH_ID_2) {
            T2 tempRes1 = Simt::UintDiv(tempRes, mH, shiftH);
            H = tempRes - tempRes1 * lenDstH;
            C = Simt::UintDiv(tempRes1, mD, shiftD);
            D = tempRes1 - C * lenDstD;
        }
        if constexpr (schId == SCH_ID_3) {
            T2 tempRes1 = Simt::UintDiv(tempRes, mH, shiftH);
            H = tempRes - tempRes1 * lenDstH;
            N = Simt::UintDiv(tempRes1, mD, shiftD);
            D = tempRes1 - N * lenDstD;
        }
        T2 origD = 0, origH = 0, origW = 0, origDUp = 0, origHUp = 0, origWUp = 0;
        ComputeOrig<T2, isExtra>(D, lenSrcD, scaleD, origD);
        ComputeOrig<T2, isExtra>(H, lenSrcH, scaleH, origH);
        ComputeOrig<T2, isExtra>(W, lenSrcW, scaleW, origW);
        ComputeOrig<T2, isExtra>(D + 1, lenSrcD, scaleD, origDUp);
        ComputeOrig<T2, isExtra>(H + 1, lenSrcH, scaleH, origHUp);
        ComputeOrig<T2, isExtra>(W + 1, lenSrcW, scaleW, origWUp);
        T2 lenSrcHw = lenSrcH * lenSrcW;
        T2 lenSrcDhw = lenSrcD * lenSrcHw;
        T2 lenDstDhw = lenDstD * lenDstH * lenDstW;
        if constexpr (schId == SCH_ID_3) {
            ComputeForSplitNcdhw<T1, T2>(
                inputGm, outputGm, lenSrcW, lenSrcHw, lenSrcDhw, N, origD, origDUp, origH, origHUp, origW, origWUp,
                yGmIdx);
        }
        if constexpr (schId == SCH_ID_1) {
            ComputeForSplitDhw<T1, T2>(
                inputGm, outputGm, lenN, lenSrcW, lenSrcHw, lenSrcDhw, lenDstDhw, origD, origDUp, origH, origHUp, origW,
                origWUp, yGmIdx);
        }
        if constexpr (schId == SCH_ID_2) {
            ComputeForSplitCdhw<T1, T2>(
                inputGm, outputGm, lenN, lenC, lenSrcW, lenSrcHw, lenSrcDhw, lenDstDhw, C, origD, origDUp, origH,
                origHUp, origW, origWUp, yGmIdx);
        }
    }
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__ void calleeInt32(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD,
    T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW)
{
    SimtCompute<T1, T2, isExtra, schId>(
        inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW);
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__ void calleeInt64(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD,
    T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW)
{
    SimtCompute<T1, T2, isExtra, schId>(
        inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW);
}
} // namespace UpsampleNearest3dGrad
#endif // UPSAMPLE_NEAREST3D_GRAD_SIMT_BASE_H