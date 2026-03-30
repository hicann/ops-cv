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
 * \file upsample_bilinear2d_aa_simt_base.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_SIMT_BASE_H
#define UPSAMPLE_BILINEAR2D_AA_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bilinear2d_aa_tiling_data.h"

namespace UpsampleBilinear2dAA {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 512;
const int32_t THREAD_NUM_B64 = 512;

static __simt_callee__ __aicore__ inline float CubilFilterAA(float x)
{
    x = Simt::Abs(x);
    if (x < 1.0f) {
        return static_cast<float>(1.0f - x);
    } 
    return 0.0f;
}

template <typename T1, typename T2, typename T3>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    T3 lenSrcHw = lenSrcH * lenSrcW;
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        T3 yGmIdx = blkStartOffset + idx;
        T2 tmpRes = Simt::UintDiv(static_cast<T2>(yGmIdx), mW, shiftW);
        T2 W = yGmIdx - tmpRes * lenDstW;
        T2 NC = Simt::UintDiv(tmpRes, mH, shiftH);
        T2 H = tmpRes - NC * lenDstH;

        const float centerH = scaleH * (static_cast<float>(H) + 0.5f);
        T3 minH = Simt::Max(static_cast<T3>(Simt::Floor(centerH - supportH + 0.5f)), static_cast<T3>(0));
        T3 maxH = Simt::Min(static_cast<T3>(Simt::Floor(centerH + supportH + 0.5f)), lenSrcH);
        float totalWeightsH = 0.0f;
        for (T3 j = minH; j < maxH; j++) {
            const float distanceH = (static_cast<float>(j) - centerH + 0.5f) * invScaleH;
            const float w = CubilFilterAA(distanceH);
            totalWeightsH += w;
        }

        const float centerW = scaleW * (static_cast<float>(W) + 0.5f);
        T3 minW = Simt::Max(static_cast<T3>(Simt::Floor(centerW - supportW + 0.5f)), static_cast<T3>(0));
        T3 maxW = Simt::Min(static_cast<T3>(Simt::Floor(centerW + supportW + 0.5f)), lenSrcW);
        float totalWeightsW = 0.0f;
        for (T3 i = minW; i < maxW; i++) {
            const float distanceW = (static_cast<float>(i) - centerW + 0.5f) * invScaleW;
            const float w = CubilFilterAA(distanceW);
            totalWeightsW += w;
        }

        float value = 0.0f;
        for (T3 h = minH; h < maxH; h++) {
            const float distanceH = (static_cast<float>(h) - centerH + 0.5f) * invScaleH;
            float weightsH = CubilFilterAA(distanceH);
            if (totalWeightsH != 0.0f) {
                weightsH /= totalWeightsH;
            }            
            for (T3 w = minW; w < maxW; w++) {
                const float distanceW = (static_cast<float>(w) - centerW + 0.5f) * invScaleW;
                float weightsW = CubilFilterAA(distanceW);
                if (totalWeightsW != 0.0f) {
                    weightsW /= totalWeightsW;
                }          
                const T3 input_idx = NC * lenSrcHw + h * lenSrcW + w;
                value += inputGm[input_idx] * weightsH * weightsW;
            }
        }
        outputGm[yGmIdx] = value;
    }
}

template <typename T1, typename T2, typename T3>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32)__aicore__ void calleeInt32(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtCompute<T1, T2, T3>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, 
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}

template <typename T1, typename T2, typename T3>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64)__aicore__ void calleeInt64(__gm__ T1 *inputGm, __gm__ T1 *outputGm, 
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtCompute<T1, T2, T3>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH,
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}
} // namespace UpsampleBilinear2dAA

#endif // UPSAMPLE_BILINEAR2D_AA_SIMT_BASE_H
