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
 * \file upsample_bicubic2d_aa_grad_simt_base.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_H
#define UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_grad_tiling_data.h"
#include "./upsample_bicubic2d_aa_grad_common.h"

namespace UpsampleBicubic2dAAGrad {
using namespace AscendC;

template <typename T2, typename T3>
static __simt_callee__ __aicore__ inline void CalculateBoundariesAndWeights(const T2 coord, const float scale, 
    const float invScale, const float support, const T3 lenDst, T3 &minCoord, 
    T3 &maxCoord, float &totalWeight)
{
    const float inputCenter = (scale > 0.0f) ? ((static_cast<float>(coord) + 0.5f) * scale) : 0.5f;
    
    minCoord = static_cast<T3>(Simt::Floor(inputCenter - support + 0.5f));
    maxCoord = static_cast<T3>(Simt::Floor(inputCenter + support + 0.5f));
    minCoord = Simt::Max(minCoord, static_cast<T3>(0));
    maxCoord = Simt::Min(maxCoord, lenDst);
    
    totalWeight = 0.0f;
    for (T3 i = minCoord; i < maxCoord; ++i) {
        totalWeight += CubicFilterAA((static_cast<float>(i) - inputCenter + 0.5f) * invScale);
    }
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T3 blkStartOffset, T3 blkProcessNum, T3 lenN, T3 lenC, T2 mH, T2 shiftH, T2 mW, T2 shiftW, T3 lenSrcH, 
    T3 lenSrcW, T3 lenDstH, T3 lenDstW, float scaleH, float scaleW, float invScaleH, float invScaleW, 
    float supportH, float supportW)
{
    for (T3 idx = static_cast<T3>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T3>(Simt::GetThreadNum<0>())) {
        const T3 xGmIdx = blkStartOffset + idx;
        const float inputValue = static_cast<float>(inputGm[xGmIdx]);
        
        // 计算W, H, NC索引
        const T2 tmpRes = Simt::UintDiv(static_cast<T2>(xGmIdx), mW, shiftW);
        const T2 W = xGmIdx - tmpRes * lenSrcW;
        const T2 NC = Simt::UintDiv(tmpRes, mH, shiftH);
        const T2 H = tmpRes - NC * lenSrcH;
        const T3 baseOutputIdx = NC * lenDstH * lenDstW;
        
        T3 minH, maxH;
        float totalWeightH;
        CalculateBoundariesAndWeights(H, scaleH, invScaleH, supportH, lenDstH, minH, maxH, totalWeightH);
        
        T3 minW, maxW;
        float totalWeightW;
        CalculateBoundariesAndWeights(W, scaleW, invScaleW, supportW, lenDstW, minW, maxW, totalWeightW);
        
        if (totalWeightH == 0.0f || totalWeightW == 0.0f) {
            continue;
        }
        
        const float inputCenterH = (scaleH > 0.0f) ? ((static_cast<float>(H) + 0.5f) * scaleH) : 0.5f;
        const float inputCenterW = (scaleW > 0.0f) ? ((static_cast<float>(W) + 0.5f) * scaleW) : 0.5f;
        for (T3 h = minH; h < maxH; ++h) {
            const float weightH = CubicFilterAA((static_cast<float>(h) - inputCenterH + 0.5f) * invScaleH) / totalWeightH;
            if (weightH == 0.0f) {
                continue;
            }
            const T3 clampedHOffset = Simt::Max(static_cast<T3>(0), Simt::Min(lenDstH, h)) * lenDstW + baseOutputIdx;
            
            for (T3 w = minW; w < maxW; ++w) {
                const float weightW = CubicFilterAA((static_cast<float>(w) - inputCenterW + 0.5f) * invScaleW) / totalWeightW;
                if (weightW == 0.0f) {
                    continue;
                }
                const T3 clampedW = Simt::Max(static_cast<T3>(0), Simt::Min(lenDstW - 1, w));
                const T3 outputIdx = clampedHOffset + clampedW;
                const float value = inputValue * weightH * weightW;
                Simt::AtomicAdd(outputGm + outputIdx, static_cast<T1>(value));
            }
        }
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

}// namespace UpsampleBicubic2dAAGrad
#endif // UPSAMPLE_BICUBIC2D_AA_GRAD_SIMT_BASE_H
