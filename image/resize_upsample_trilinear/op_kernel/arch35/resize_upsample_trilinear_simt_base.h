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

#include "./resize_upsample_trilinear_tiling_data.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

constexpr int32_t THREAD_NUM_B32 = 512;
constexpr int32_t THREAD_NUM_B64 = 512;
constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t INTERP_NEIGHBOR_SIZE = 2;

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeSourceIndex(float scale, T2 dstIndex,
                                                                                          int32_t alignCorners)
{
    if (alignCorners == 1) {
        return fmaf(static_cast<float>(dstIndex), scale, 0.0f);
    }
    float srcIndex = fmaf(static_cast<float>(dstIndex) + HALF_PIXEL, scale, -HALF_PIXEL);
    return srcIndex < 0.0f ? 0.0f : srcIndex;
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeLinearIndexAndWeight(
    float realIndex, T2 limit, T2& index0, T2& index1, float& weight0, float& weight1)
{
    index0 = min(static_cast<T2>(realIndex), limit);
    index1 = min(static_cast<T2>(index0 + 1), limit);
    weight1 = min(max(realIndex - static_cast<float>(index0), 0.0f), 1.0f);
    weight0 = 1.0f - weight1;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool IsZeroWeightSpecialValue(float inputValue,
                                                                                               float weightD,
                                                                                               float weightH,
                                                                                               float weightW)
{
    return (weightD == 0.0f || weightH == 0.0f || weightW == 0.0f) && (isinf(inputValue) || isnan(inputValue));
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeOutIndex(T2 yGmIdx, T2 mW, T2 shiftW,
                                                                                      T2 lenDstW, T2 mH, T2 shiftH,
                                                                                      T2 lenDstH, T2 mD, T2 shiftD,
                                                                                      T2 lenDstD, T2& outW, T2& outH,
                                                                                      T2& outD, T2& nc)
{
    T2 tmpW = Simt::UintDiv(yGmIdx, mW, shiftW);
    outW = yGmIdx - tmpW * lenDstW;
    T2 tmpH = Simt::UintDiv(tmpW, mH, shiftH);
    outH = tmpW - tmpH * lenDstH;
    nc = Simt::UintDiv(tmpH, mD, shiftD);
    outD = tmpH - nc * lenDstD;
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeInterpParams(
    float scaleD, T2 outD, T2 lenSrcD, float scaleH, T2 outH, T2 lenSrcH, float scaleW, T2 outW, T2 lenSrcW,
    int32_t alignCorners, T2& inD0, T2& inD1, T2& inH0, T2& inH1, T2& inW0, T2& inW1, float& weightD0, float& weightD1,
    float& weightH0, float& weightH1, float& weightW0, float& weightW1)
{
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleD, outD, alignCorners), lenSrcD - 1, inD0, inD1, weightD0,
                                weightD1);
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleH, outH, alignCorners), lenSrcH - 1, inH0, inH1, weightH0,
                                weightH1);
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleW, outW, alignCorners), lenSrcW - 1, inW0, inW1, weightW0,
                                weightW1);
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void LoadInputValues(
    __gm__ T1* inputGm, T2 ncOffset, T2 inD0, T2 inD1, T2 inH0, T2 inH1, T2 inW0, T2 inW1, T2 lenSrcHw, T2 lenSrcW,
    float& value000, float& value001, float& value010, float& value011, float& value100, float& value101,
    float& value110, float& value111)
{
    T2 d0Offset = ncOffset + inD0 * lenSrcHw;
    T2 d1Offset = ncOffset + inD1 * lenSrcHw;
    T2 h00Offset = d0Offset + inH0 * lenSrcW;
    T2 h01Offset = d0Offset + inH1 * lenSrcW;
    T2 h10Offset = d1Offset + inH0 * lenSrcW;
    T2 h11Offset = d1Offset + inH1 * lenSrcW;
    value000 = static_cast<float>(inputGm[h00Offset + inW0]);
    value001 = static_cast<float>(inputGm[h00Offset + inW1]);
    value010 = static_cast<float>(inputGm[h01Offset + inW0]);
    value011 = static_cast<float>(inputGm[h01Offset + inW1]);
    value100 = static_cast<float>(inputGm[h10Offset + inW0]);
    value101 = static_cast<float>(inputGm[h10Offset + inW1]);
    value110 = static_cast<float>(inputGm[h11Offset + inW0]);
    value111 = static_cast<float>(inputGm[h11Offset + inW1]);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool CheckSpecialZeroWeight(
    float value000, float value001, float value010, float value011, float value100, float value101, float value110,
    float value111, float weightD0, float weightD1, float weightH0, float weightH1, float weightW0, float weightW1)
{
    return IsZeroWeightSpecialValue(value000, weightD0, weightH0, weightW0) ||
           IsZeroWeightSpecialValue(value001, weightD0, weightH0, weightW1) ||
           IsZeroWeightSpecialValue(value010, weightD0, weightH1, weightW0) ||
           IsZeroWeightSpecialValue(value011, weightD0, weightH1, weightW1) ||
           IsZeroWeightSpecialValue(value100, weightD1, weightH0, weightW0) ||
           IsZeroWeightSpecialValue(value101, weightD1, weightH0, weightW1) ||
           IsZeroWeightSpecialValue(value110, weightD1, weightH1, weightW0) ||
           IsZeroWeightSpecialValue(value111, weightD1, weightH1, weightW1);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeTrilinearValue(
    float weightD0, float weightD1, float weightH0, float weightH1, float weightW0, float weightW1, float value000,
    float value001, float value010, float value011, float value100, float value101, float value110, float value111)
{
    return weightD0 * (weightH0 * (weightW0 * value000 + weightW1 * value001) +
                       weightH1 * (weightW0 * value010 + weightW1 * value011)) +
           weightD1 * (weightH0 * (weightW0 * value100 + weightW1 * value101) +
                       weightH1 * (weightW0 * value110 + weightW1 * value111));
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD,
    float scaleH, float scaleW, int32_t alignCorners)
{
    T2 lenSrcHw = lenSrcH * lenSrcW;
    T2 lenSrcDhw = lenSrcD * lenSrcHw;
    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += static_cast<T2>(blockDim.x)) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 outW = 0;
        T2 outH = 0;
        T2 outD = 0;
        T2 nc = 0;
        ComputeOutIndex(yGmIdx, mW, shiftW, lenDstW, mH, shiftH, lenDstH, mD, shiftD, lenDstD, outW, outH, outD, nc);

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
        ComputeInterpParams(scaleD, outD, lenSrcD, scaleH, outH, lenSrcH, scaleW, outW, lenSrcW, alignCorners, inD0,
                            inD1, inH0, inH1, inW0, inW1, weightD0, weightD1, weightH0, weightH1, weightW0, weightW1);

        T2 ncOffset = nc * lenSrcDhw;
        float value000 = 0.0f;
        float value001 = 0.0f;
        float value010 = 0.0f;
        float value011 = 0.0f;
        float value100 = 0.0f;
        float value101 = 0.0f;
        float value110 = 0.0f;
        float value111 = 0.0f;
        LoadInputValues<T1, T2>(inputGm, ncOffset, inD0, inD1, inH0, inH1, inW0, inW1, lenSrcHw, lenSrcW, value000,
                                value001, value010, value011, value100, value101, value110, value111);

        bool specialZeroWeight = CheckSpecialZeroWeight(value000, value001, value010, value011, value100, value101,
                                                        value110, value111, weightD0, weightD1, weightH0, weightH1,
                                                        weightW0, weightW1);
        float value = ASCRT_NAN_F;
        if (!specialZeroWeight) {
            value = ComputeTrilinearValue(weightD0, weightD1, weightH0, weightH1, weightW0, weightW1, value000,
                                          value001, value010, value011, value100, value101, value110, value111);
        }
        outputGm[yGmIdx] = static_cast<T1>(value);
    }
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__
    void calleeInt32(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH,
                     T2 lenDstW, float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
                        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__
    void calleeInt64(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH,
                     T2 lenDstW, float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
                        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_