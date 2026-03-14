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
 * \file upsample_bicubic2d_aa_grad_simt_base_deter.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_DETER_H
#define UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_DETER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_grad_tiling_data.h"
#include "./upsample_bicubic2d_aa_grad_common.h"

namespace UpsampleBicubic2dAAGrad {
using namespace AscendC;

template <typename T3>
static __aicore__ inline void CalculateBounds(T3 &minVal, T3 &maxVal, float center, float scale, float support, T3 lenSrc)
{
    if (likely(scale > 0.0f)) {
        minVal = static_cast<T3>(Simt::Floor((center - support) / scale + 0.5f));
        maxVal = static_cast<T3>(Simt::Floor((center + support) / scale - 0.5f)) + 1;
        minVal = Simt::Max(minVal, static_cast<T3>(0));
        maxVal = Simt::Min(maxVal, lenSrc);
    } else {
        minVal = 0;
        maxVal = lenSrc;
    }
}

template <typename T3>
static __aicore__ inline float CalculateWeight(float center, float outputCenter, float invScale, float support, T3 lenDst)
{
    float weight = CubicFilterAA((center - outputCenter) * invScale);
    if (weight == 0.0f) {
        return 0.0f;
    }

    float totalWeight = 0.0f;
    T3 minDst = static_cast<T3>(Simt::Floor(outputCenter - support - 0.5f));
    T3 maxDst = static_cast<T3>(Simt::Floor(outputCenter + support + 0.5f)) + 1;
    minDst = Simt::Max(minDst, static_cast<T3>(0));
    maxDst = Simt::Min(maxDst, lenDst);

    for (T3 k = minDst; k < maxDst; ++k) {
        float outCenter = static_cast<float>(k) + 0.5f;
        if (Simt::Abs(outCenter - outputCenter) > support) {
            continue;
        }
        totalWeight += CubicFilterAA((outCenter - outputCenter) * invScale);
    }

    if (totalWeight != 0.0f) {
        weight /= totalWeight;
    }
    return weight;
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__aicore__ __attribute__((always_inline)) inline void SimtDeterCompute(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        T3 yGmIdx = blkStartOffset + idx;
        T2 W = 0;
        T2 H = 0;
        T2 Batch = 0;
        T2 tmpRes = Simt::UintDiv(static_cast<T2>(yGmIdx), mW, shiftW);
        W = yGmIdx - tmpRes * lenDstW;
        Batch = Simt::UintDiv(tmpRes, mH, shiftH);
        H = tmpRes - Batch * lenDstH;

        // 计算中心位置
        float centerH = static_cast<float>(H) + 0.5f;
        float centerW = static_cast<float>(W) + 0.5f;

        T3 minH, maxH, minW, maxW;
        CalculateBounds(minH, maxH, centerH, scaleH, supportH, lenSrcH);
        CalculateBounds(minW, maxW, centerW, scaleW, supportW, lenSrcW);

        float value = 0.0f;
        T3 lenSrcHw = lenSrcH * lenSrcW;

        for (T3 h = minH; h < maxH; ++h) {
            const float outputCenterH = (static_cast<float>(h) + 0.5f) * scaleH;
            float weightH = CalculateWeight(centerH, outputCenterH, invScaleH, supportH, lenDstH);
            if (weightH == 0.0f) {
                continue;
            }

            const T3 clampedH = Simt::Max(static_cast<T3>(0), Simt::Min(lenSrcH, h));               
            for (T3 w = minW; w < maxW; ++w) {
                const float outputCenterW = (static_cast<float>(w) + 0.5f) * scaleW;
                float weightW = CalculateWeight(centerW, outputCenterW, invScaleW, supportW, lenDstW);
                if (weightW == 0.0f) {
                    continue;
                }

                const T3 clampedW = Simt::Max(static_cast<T3>(0), Simt::Min(lenSrcW - 1, w));              
                const T3 input_idx = Batch * lenSrcHw + clampedH * lenSrcW + clampedW;
                value += inputGm[input_idx] * weightH * weightW;
            }
        }
        outputGm[yGmIdx] = value;
    }
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32)__aicore__ void calleeDeterInt32(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtDeterCompute<T1, T2, T3, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, 
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64)__aicore__ void calleeDeterInt64(__gm__ T1 *inputGm, __gm__ T1 *outputGm, 
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    SimtDeterCompute<T1, T2, T3, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH,
        mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, invScaleH, invScaleW, supportH, supportW);
}

}// namespace UpsampleBicubic2dAAGrad
#endif // UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_DETER_H