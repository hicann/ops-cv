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
 * \file resize_bicubic_v2_simt_base.h
 * \brief resize_bicubic_v2_simt_base
 */
#ifndef RESIZE_BICUBIC_v2_SIMT_BASE_H
#define RESIZE_BICUBIC_v2_SIMT_BASE_H

#include "kernel_operator.h"

namespace ResizeBicubicV2 {
using namespace AscendC;
static __aicore__ inline float CubicConvolution2(float x, float a)
{
    return static_cast<float>(((a * x - 5 * a) * x + 8 * a) * x - 4 * a);
}

static __aicore__ inline float CubicConvolution1(float x, float a)
{
    return static_cast<float>(((a + 2) * x - (a + 3)) * x * x + 1);
}

static __aicore__ inline void GetCubicCoeff(float lep, float &cof0, float &cof1, float &cof2, float &cof3)
{
    float A = -0.75f;
    cof0 = CubicConvolution2((lep + 1.0f), A);
    cof1 = CubicConvolution1(lep, A);
    cof2 = CubicConvolution1(1.0f - lep, A);
    cof3 = CubicConvolution2(2.0f - lep, A);
}

template <typename T_IDX2>
static __aicore__ inline T_IDX2 GetSrc(T_IDX2 src, T_IDX2 maxLimt)
{
    if (src < 0) {
        src = 0;
    }
    if (src > maxLimt) {
        src = maxLimt;
    }
    return src;
}

template <typename T_IDX, uint64_t halfPixel>
__aicore__ __attribute__((always_inline)) inline float ComputeOri(T_IDX H, float scaleH)
{
    if constexpr (halfPixel == 1) {
        float orig = static_cast<float>((H + 0.5f) * scaleH) - 0.5f;
        return orig;
    } else {
        float orig = static_cast<float>(H * scaleH);
        return orig;
    }
}

template <typename T1, typename T_IDX, typename T_IDX2>
__aicore__ __attribute__((always_inline)) inline void ComputeNchwMode0(T_IDX origBaseIdx, T_IDX yGmIdx, T_IDX lenSrcW,
    float origHeight, float origWidth, T_IDX2 lenSrcH1, T_IDX2 lenSrcW1, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    T_IDX2 leftX = Simt::Floor(origWidth);
    T_IDX2 topY = Simt::Floor(origHeight);
    float deltaX = origWidth - static_cast<float>(leftX);
    float deltaY = origHeight - static_cast<float>(topY);
    float coffW0 = 0.0f;
    float coffW1 = 0.0f;
    float coffW2 = 0.0f;
    float coffW3 = 0.0f;
    float coffH0 = 0.0f;
    float coffH1 = 0.0f;
    float coffH2 = 0.0f;
    float coffH3 = 0.0f;
    GetCubicCoeff(deltaX, coffW0, coffW1, coffW2, coffW3);
    GetCubicCoeff(deltaY, coffH0, coffH1, coffH2, coffH3);
    T_IDX2 h0 = GetSrc<T_IDX2>(topY - 1, lenSrcH1);
    T_IDX2 h1 = GetSrc<T_IDX2>(topY, lenSrcH1);
    T_IDX2 h2 = GetSrc<T_IDX2>(topY + 1, lenSrcH1);
    T_IDX2 h3 = GetSrc<T_IDX2>(topY + 2, lenSrcH1);
    T_IDX2 w0 = GetSrc<T_IDX2>(leftX - 1, lenSrcW1);
    T_IDX2 w1 = GetSrc<T_IDX2>(leftX, lenSrcW1);
    T_IDX2 w2 = GetSrc<T_IDX2>(leftX + 1, lenSrcW1);
    T_IDX2 w3 = GetSrc<T_IDX2>(leftX + 2, lenSrcW1);
    T_IDX inputGmOffset0 = origBaseIdx + h0 * lenSrcW;
    T_IDX inputGmOffset1 = origBaseIdx + h1 * lenSrcW;
    T_IDX inputGmOffset2 = origBaseIdx + h2 * lenSrcW;
    T_IDX inputGmOffset3 = origBaseIdx + h3 * lenSrcW;
    float value0 = static_cast<float>(inputGm[inputGmOffset0 + w0]);
    float value1 = static_cast<float>(inputGm[inputGmOffset0 + w1]);
    float value2 = static_cast<float>(inputGm[inputGmOffset0 + w2]);
    float value3 = static_cast<float>(inputGm[inputGmOffset0 + w3]);
    float valueW0 = value0 * coffW0 + value1 * coffW1 + value2 * coffW2 + value3 * coffW3;
    float value00 = static_cast<float>(inputGm[inputGmOffset1 + w0]);
    float value10 = static_cast<float>(inputGm[inputGmOffset1 + w1]);
    float value20 = static_cast<float>(inputGm[inputGmOffset1 + w2]);
    float value30 = static_cast<float>(inputGm[inputGmOffset1 + w3]);
    float valueW1 = value00 * coffW0 + value10 * coffW1 + value20 * coffW2 + value30 * coffW3;
    float value01 = static_cast<float>(inputGm[inputGmOffset2 + w0]);
    float value11 = static_cast<float>(inputGm[inputGmOffset2 + w1]);
    float value21 = static_cast<float>(inputGm[inputGmOffset2 + w2]);
    float value31 = static_cast<float>(inputGm[inputGmOffset2 + w3]);
    float valueW2 = value01 * coffW0 + value11 * coffW1 + value21 * coffW2 + value31 * coffW3;
    float value02 = static_cast<float>(inputGm[inputGmOffset3 + w0]);
    float value12 = static_cast<float>(inputGm[inputGmOffset3 + w1]);
    float value22 = static_cast<float>(inputGm[inputGmOffset3 + w2]);
    float value32 = static_cast<float>(inputGm[inputGmOffset3 + w3]);
    float valueW3 = value02 * coffW0 + value12 * coffW1 + value22 * coffW2 + value32 * coffW3;
    float valueDst = valueW0 * coffH0 + valueW1 * coffH1 + valueW2 * coffH2 + valueW3 * coffH3;
    outputGm[yGmIdx] = static_cast<T1>(valueDst);
}

template <typename T1, typename T_IDX, typename T_IDX2>
__aicore__ __attribute__((always_inline)) inline void ComputeNhwcMode0(T_IDX yGmIdx, float origWidth, float origHeight,
    T_IDX origBaseIdx, T_IDX2 lenSrcH1, T_IDX2 lenSrcW1, T_IDX lenC, T_IDX lenSrcWc, __gm__ T1 *inputGm,
    __gm__ T1 *outputGm)
{
    T_IDX2 leftX = Simt::Floor(origWidth);
    T_IDX2 topY = Simt::Floor(origHeight);
    float deltaX = origWidth - static_cast<float>(leftX);
    float deltaY = origHeight - static_cast<float>(topY);
    float coffW0 = 0.0f;
    float coffW1 = 0.0f;
    float coffW2 = 0.0f;
    float coffW3 = 0.0f;
    float coffH0 = 0.0f;
    float coffH1 = 0.0f;
    float coffH2 = 0.0f;
    float coffH3 = 0.0f;
    GetCubicCoeff(deltaX, coffW0, coffW1, coffW2, coffW3);
    GetCubicCoeff(deltaY, coffH0, coffH1, coffH2, coffH3);
    T_IDX inputGmOffset0 = origBaseIdx + GetSrc<T_IDX2>(topY - 1, lenSrcH1) * lenSrcWc;
    T_IDX inputGmOffset1 = origBaseIdx + GetSrc<T_IDX2>(topY, lenSrcH1) * lenSrcWc;
    T_IDX inputGmOffset2 = origBaseIdx + GetSrc<T_IDX2>(topY + 1, lenSrcH1) * lenSrcWc;
    T_IDX inputGmOffset3 = origBaseIdx + GetSrc<T_IDX2>(topY + 2, lenSrcH1) * lenSrcWc;
    T_IDX wOffset0 = GetSrc<T_IDX2>(leftX - 1, lenSrcW1) * lenC;
    T_IDX wOffset1 = GetSrc<T_IDX2>(leftX, lenSrcW1) * lenC;
    T_IDX wOffset2 = GetSrc<T_IDX2>(leftX + 1, lenSrcW1) * lenC;
    T_IDX wOffset3 = GetSrc<T_IDX2>(leftX + 2, lenSrcW1) * lenC;
    float value0 = static_cast<float>(inputGm[inputGmOffset0 + wOffset0]);
    float value1 = static_cast<float>(inputGm[inputGmOffset0 + wOffset1]);
    float value2 = static_cast<float>(inputGm[inputGmOffset0 + wOffset2]);
    float value3 = static_cast<float>(inputGm[inputGmOffset0 + wOffset3]);
    float valueW0 = value0 * coffW0 + value1 * coffW1 + value2 * coffW2 + value3 * coffW3;
    float value00 = static_cast<float>(inputGm[inputGmOffset1 + wOffset0]);
    float value10 = static_cast<float>(inputGm[inputGmOffset1 + wOffset1]);
    float value20 = static_cast<float>(inputGm[inputGmOffset1 + wOffset2]);
    float value30 = static_cast<float>(inputGm[inputGmOffset1 + wOffset3]);
    float valueW1 = value00 * coffW0 + value10 * coffW1 + value20 * coffW2 + value30 * coffW3;
    float value01 = static_cast<float>(inputGm[inputGmOffset2 + wOffset0]);
    float value11 = static_cast<float>(inputGm[inputGmOffset2 + wOffset1]);
    float value21 = static_cast<float>(inputGm[inputGmOffset2 + wOffset2]);
    float value31 = static_cast<float>(inputGm[inputGmOffset2 + wOffset3]);
    float valueW2 = value01 * coffW0 + value11 * coffW1 + value21 * coffW2 + value31 * coffW3;
    float value02 = static_cast<float>(inputGm[inputGmOffset3 + wOffset0]);
    float value12 = static_cast<float>(inputGm[inputGmOffset3 + wOffset1]);
    float value22 = static_cast<float>(inputGm[inputGmOffset3 + wOffset2]);
    float value32 = static_cast<float>(inputGm[inputGmOffset3 + wOffset3]);
    float valueW3 = value02 * coffW0 + value12 * coffW1 + value22 * coffW2 + value32 * coffW3;
    float valueDst = valueW0 * coffH0 + valueW1 * coffH1 + valueW2 * coffH2 + valueW3 * coffH3;
    outputGm[yGmIdx] = static_cast<T1>(valueDst);
}
}  // namespace ResizeBicubicV2
#endif  // RESIZE_NEAREAST_NEIGHBOR_V2_BASE_H
