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
 * \file resize_upsample_trilinear_simt_base.h
 * \brief ResizeUpsampleTrilinear SIMT compute base for arch35.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

constexpr int32_t THREAD_NUM_B32 = 512;
constexpr int32_t THREAD_NUM_B64 = 512;
constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t INTERP_NEIGHBOR_SIZE = 2;
constexpr float MAX_SAFE_UINT32_INDEX = 4294967040.0f;
constexpr float MAX_SAFE_UINT64_INDEX = 18446742974197923840.0f;

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeSourceIndex(
    float scale, T2 dstIndex, int32_t alignCorners)
{
    if (alignCorners == 1) {
        return scale * static_cast<float>(dstIndex);
    }
    float srcIndex = scale * (static_cast<float>(dstIndex) + HALF_PIXEL) - HALF_PIXEL;
    return srcIndex < 0.0f ? 0.0f : srcIndex;
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeLinearIndexAndWeight(
    float realIndex, T2 limit, T2& index0, T2& index1, float& weight0, float& weight1)
{
    if (!(realIndex > 0.0f)) {
        index0 = 0;
    } else {
        float maxSafeIndex = sizeof(T2) == sizeof(uint32_t) ? MAX_SAFE_UINT32_INDEX : MAX_SAFE_UINT64_INDEX;
        index0 = realIndex > maxSafeIndex ? limit : min(static_cast<T2>(realIndex), limit);
    }
    index1 = index0 >= limit ? limit : index0 + static_cast<T2>(1);
    weight1 = min(max(realIndex - static_cast<float>(index0), 0.0f), 1.0f);
    weight0 = 1.0f - weight1;
    if (index0 == index1) {
        weight0 = 1.0f;
        weight1 = 0.0f;
    }
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH,
    T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    T2 lenSrcHw = lenSrcH * lenSrcW;
    T2 lenSrcDhw = lenSrcD * lenSrcHw;
    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += static_cast<T2>(blockDim.x)) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 tmpW = Simt::UintDiv(yGmIdx, mW, shiftW);
        T2 outW = yGmIdx - tmpW * lenDstW;
        T2 tmpH = Simt::UintDiv(tmpW, mH, shiftH);
        T2 outH = tmpW - tmpH * lenDstH;
        T2 nc = Simt::UintDiv(tmpH, mD, shiftD);
        T2 outD = tmpH - nc * lenDstD;

        T2 inD0 = 0;
        T2 inD1 = 0;
        T2 inH0 = 0;
        T2 inH1 = 0;
        T2 inW0 = 0;
        T2 inW1 = 0;
        float weightD0 = 0.0f;
        float weightD1 = 0.0f;
        float weightH0 = 0.0f;
        float weightH1 = 0.0f;
        float weightW0 = 0.0f;
        float weightW1 = 0.0f;

        ComputeLinearIndexAndWeight(
            ComputeSourceIndex(scaleD, outD, alignCorners), lenSrcD - 1, inD0, inD1, weightD0, weightD1);
        ComputeLinearIndexAndWeight(
            ComputeSourceIndex(scaleH, outH, alignCorners), lenSrcH - 1, inH0, inH1, weightH0, weightH1);
        ComputeLinearIndexAndWeight(
            ComputeSourceIndex(scaleW, outW, alignCorners), lenSrcW - 1, inW0, inW1, weightW0, weightW1);

        T2 ncOffset = nc * lenSrcDhw;
        float value = 0.0f;
        for (T2 d = 0; d < static_cast<T2>(INTERP_NEIGHBOR_SIZE); d++) {
            T2 srcD = (d == 0) ? inD0 : inD1;
            float weightD = (d == 0) ? weightD0 : weightD1;
            for (T2 h = 0; h < static_cast<T2>(INTERP_NEIGHBOR_SIZE); h++) {
                T2 srcH = (h == 0) ? inH0 : inH1;
                float weightH = (h == 0) ? weightH0 : weightH1;
                for (T2 w = 0; w < static_cast<T2>(INTERP_NEIGHBOR_SIZE); w++) {
                    T2 srcW = (w == 0) ? inW0 : inW1;
                    float weightW = (w == 0) ? weightW0 : weightW1;
                    T2 inputOffset = ncOffset + srcD * lenSrcHw + srcH * lenSrcW + srcW;
                    value += static_cast<float>(inputGm[inputOffset]) * weightD * weightH * weightW;
                }
            }
        }
        outputGm[yGmIdx] = static_cast<T1>(value);
    }
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__ void calleeInt32(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH,
    T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(
        inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH,
        lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__ void calleeInt64(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH,
    T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
    float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(
        inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH,
        lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_
