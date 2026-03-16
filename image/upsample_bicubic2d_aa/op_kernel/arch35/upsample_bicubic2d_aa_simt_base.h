/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bicubic2d_aa_simt_base.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_AA_SIMT_BASE_H
#define UPSAMPLE_BICUBIC2D_AA_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_tiling_data.h"

namespace UpsampleBicubic2dAA {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 512;
const int32_t THREAD_NUM_B64 = 512;
const uint64_t SCH_ID_1 = 1;

static __aicore__ inline float CubicConvolution1(float x)
{
    return static_cast<float>((1.5f * x - 2.5f) * x * x + 1.0f);
}
static __aicore__ inline float CubicConvolution2(float x)
{
    return static_cast<float>(((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f);
}
static __aicore__ inline float CubicFilterAA(float x)
{
    x = Simt::Abs(x);
    if (x < 1.0f) {
        return CubicConvolution1(x);
    } else if (x < 2.0f) {
        return CubicConvolution2(x);
    } else {
        return 0.0f;
    }
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        T3 yGmIdx = blkStartOffset + idx;
        T2 W = 0;
        T2 H = 0;
        T2 NC = 0;
        T2 tmpRes = Simt::UintDiv(static_cast<T2>(yGmIdx), mW, shiftW);
        W = yGmIdx - tmpRes * lenDstW;
        NC = Simt::UintDiv(tmpRes, mH, shiftH);
        H = tmpRes - NC * lenDstH;

        // 计算中心位置
        const float centerH = scaleH * (static_cast<float>(H) + 0.5f);
        const float centerW = scaleW * (static_cast<float>(W) + 0.5f);
    
        // 计算垂直边界
        T3 minH = static_cast<T3>(Simt::Floor(centerH - supportH + 0.5f));
        T3 maxH = static_cast<T3>(Simt::Floor(centerH + supportH + 0.5f));
        minH = Simt::Max(minH, static_cast<T3>(0));
        maxH = Simt::Min(maxH, lenSrcH);
        const T3 sizeH = maxH - minH;
    
        // 计算水平边界
        T3 minW = static_cast<T3>(Simt::Floor(centerW - supportW + 0.5f));
        T3 maxW = static_cast<T3>(Simt::Floor(centerW + supportW + 0.5f));
        minW = Simt::Max(minW, static_cast<T3>(0));
        maxW = Simt::Min(maxW, lenSrcW);
        const T3 sizeW = maxW - minW;
    
        // 计算垂直权重和
        float totalWeightsH = 0.0f;
        for (T3 j = 0; j < sizeH; ++j) {
            const T3 jH = minH + j;
            const float distanceH = (static_cast<float>(jH) - centerH + 0.5f) * invScaleH;
            const float w = CubicFilterAA(distanceH);
            totalWeightsH += w;
        }
    
        // 计算水平权重和
        float totalWeightsW = 0.0f;
        for (T3 k = 0; k < sizeW; ++k) {
            const T3 kW = minW + k;
            const float distanceW = (static_cast<float>(kW) - centerW + 0.5f) * invScaleW;
            const float w = CubicFilterAA(distanceW);
            totalWeightsW += w;
        }

        T3 lenSrcHw = lenSrcH * lenSrcW;
        float value = 0.0f;
        for (T3 m = 0; m < sizeH; ++m) {
            const T3 mH = minH + m;
            const float distanceH = (static_cast<float>(mH) - centerH + 0.5f) * invScaleH;
            float weightsH = CubicFilterAA(distanceH);
            if (totalWeightsH != 0.0f) {
                weightsH /= totalWeightsH;
            }
            const T3 clampedH = Simt::Max(static_cast<T3>(0), Simt::Min(lenSrcH - 1, mH));               
            for (T3 n = 0; n < sizeW; ++n) {
                const T3 nW = minW + n;
                const float distanceW = (static_cast<float>(nW) - centerW + 0.5f) * invScaleW;
                float weightsW = CubicFilterAA(distanceW);
                if (totalWeightsW != 0.0f) {
                    weightsW /= totalWeightsW;
                }
                const T3 clampedW = Simt::Max(static_cast<T3>(0), Simt::Min(lenSrcW - 1, nW));              
                const T3 input_idx = NC * lenSrcHw + clampedH * lenSrcW + clampedW;
                value += inputGm[input_idx] * weightsH * weightsW;
            }
        }
        outputGm[yGmIdx] = value;
    }
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32)__aicore__ void calleeInt32(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtCompute<T1, T2, T3, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, 
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64)__aicore__ void calleeInt64(__gm__ T1 *inputGm, __gm__ T1 *outputGm, 
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtCompute<T1, T2, T3, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH,
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}

}// namespace UpsampleBicubic2dAA
#endif // UPSAMPLE_BICUBIC2D_AA_SIMT_BASE_H