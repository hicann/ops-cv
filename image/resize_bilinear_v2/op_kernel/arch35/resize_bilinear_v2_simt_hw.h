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
 * \file resize_bilinear_v2_simt_hw.h
 * \brief resize_bilinear_v2_simt_hw
 */

#ifndef CANN_RESIZE_BILINEAR_V2_SIMT_HW_H
#define CANN_RESIZE_BILINEAR_V2_SIMT_HW_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_bilinear_v2_base.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
class ResizeBilinearV2SimtHW {
public:
    __aicore__ inline ResizeBilinearV2SimtHW(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR size, GM_ADDR y, const ResizeBilinearV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    const ResizeBilinearV2TilingData* tiling_;
    int64_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T2> outputGm_;
};

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
__aicore__ inline void ResizeBilinearV2SimtHW<T1, T2, halfPixel, mode, T_IDX>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeBilinearV2TilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1*)x);
    outputGm_.SetGlobalBuffer((__gm__ T2*)y);
}

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(
    float scaleH, float scaleW, T_IDX lenN, T_IDX lenC, T_IDX lenDesH, T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* input, __gm__ volatile T2* output, T_IDX blkStartOffset,
    T_IDX blkProcessNum)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yGmIdx = blkStartOffset + idx;
        T_IDX tmp = yGmIdx;

        T_IDX H = 0;
        T_IDX W = 0;

        T_IDX tmpRes = Simt::UintDiv(tmp, mW, shiftW);
        W = tmp - tmpRes * lenDesW;
        tmp = tmpRes;

        tmpRes = Simt::UintDiv(tmp, mH, shiftH);
        H = tmp - tmpRes * lenDesH;

        float origHeight = 0.0f, origWidth = 0.0f, deltaX = 0.0f, deltaY = 0.0f;
        T_IDX leftX = 0, rightX = 0, topY = 0, bottomY = 0;
        T_IDX lenSrcWH = lenSrcW * lenSrcH;
        T_IDX lenDesWH = lenDesW * lenDesH;
        float offset = (halfPixel) ? HALF_PIXEL : 0.0f;

        origHeight = static_cast<float>((H + offset) * scaleH) - offset;
        origWidth = static_cast<float>((W + offset) * scaleW) - offset;
        leftX = Simt::Max(Simt::Min(static_cast<T_IDX>(Simt::Floor(origWidth)), lenSrcW - 1), static_cast<T_IDX>(0));
        deltaX = origWidth - Simt::Floor(origWidth);
        rightX = Simt::Max(Simt::Min(static_cast<T_IDX>(Simt::Ceil(origWidth)), lenSrcW - 1), static_cast<T_IDX>(0));
        topY = Simt::Max(Simt::Min(static_cast<T_IDX>(Simt::Floor(origHeight)), lenSrcH - 1), static_cast<T_IDX>(0));
        deltaY = origHeight - Simt::Floor(origHeight);
        bottomY = Simt::Max(Simt::Min(static_cast<T_IDX>(Simt::Ceil(origHeight)), lenSrcH - 1), static_cast<T_IDX>(0));

        for (T_IDX n = 0; n < lenN; n++) {
            for (T_IDX c = 0; c < lenC; c++) {
                T_IDX origBaseIdx = n * (lenC * lenSrcWH) + c * lenSrcWH;
                float pixelLeftTop = input[origBaseIdx + topY * lenSrcW + leftX];
                float pixelRightTop = input[origBaseIdx + topY * lenSrcW + rightX];
                float pixelLeftBottom = input[origBaseIdx + bottomY * lenSrcW + leftX];
                float pixelRightBottom = input[origBaseIdx + bottomY * lenSrcW + rightX];
                T_IDX outputGmIdx = n * (lenC * lenDesWH) + c * lenDesWH + yGmIdx;
                output[outputGmIdx] = (1.0f - deltaY) * ((1.0f - deltaX) * pixelLeftTop + deltaX * pixelRightTop) +
                                      deltaY * ((1.0f - deltaX) * pixelLeftBottom + deltaX * pixelRightBottom);
            }
        }
    }
}

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void calleeInt32(
    float scaleH, float scaleW, T_IDX lenN, T_IDX lenC, T_IDX lenDesH, T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* input, __gm__ volatile T2* output, T_IDX blkStartOffset,
    T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, mode, T_IDX>(
        scaleH, scaleW, lenN, lenC, lenDesH, lenDesW, lenSrcH, lenSrcW, shiftH, mH, shiftW, mW, input, output,
        blkStartOffset, blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_MIDDLE) __aicore__ void calleeInt64(
    float scaleH, float scaleW, T_IDX lenN, T_IDX lenC, T_IDX lenDesH, T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* input, __gm__ volatile T2* output, T_IDX blkStartOffset,
    T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, mode, T_IDX>(
        scaleH, scaleW, lenN, lenC, lenDesH, lenDesW, lenSrcH, lenSrcW, shiftH, mH, shiftW, mW, input, output,
        blkStartOffset, blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, int mode, typename T_IDX>
__aicore__ inline void ResizeBilinearV2SimtHW<T1, T2, halfPixel, mode, T_IDX>::Process()
{
    T_IDX mW = 1;
    T_IDX mH = 1;
    T_IDX shiftW = 0;
    T_IDX shiftH = 0;
    GetUintDivMagicAndShift(mW, shiftW, static_cast<T_IDX>(tiling_->lenDesW));
    GetUintDivMagicAndShift(mH, shiftH, static_cast<T_IDX>(tiling_->lenDesH));

    T_IDX blkProcessNum = tiling_->splitBlockFactor;
    T_IDX blkStartOffset = blockIdx_ * tiling_->splitBlockFactor;
    if (blockIdx_ < tiling_->splitBlockTailFactor) {
        blkProcessNum += 1;
        blkStartOffset += blockIdx_;
    } else {
        blkStartOffset += tiling_->splitBlockTailFactor;
    }
    const float scaleH = tiling_->scaleH;
    const float scaleW = tiling_->scaleW;
    const T_IDX lenSrcW = tiling_->lenSrcW;
    const T_IDX lenSrcH = tiling_->lenSrcH;
    const T_IDX lenC = tiling_->lenC;
    const T_IDX lenDesW = tiling_->lenDesW;
    const T_IDX lenDesH = tiling_->lenDesH;
    const T_IDX lenN = tiling_->lenN;
    const int32_t realCoreNum = tiling_->realCoreNum;
    if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, T2, halfPixel, mode, T_IDX>>(
            Simt::Dim3(THREAD_NUM_MIDDLE), scaleH, scaleW, lenN, lenC, lenDesH, lenDesW, lenSrcH, lenSrcW, shiftH, mH,
            shiftW, mW, (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ volatile T2*)(outputGm_.GetPhyAddr()),
            blkStartOffset, blkProcessNum);
    } else {
        Simt::VF_CALL<calleeInt32<T1, T2, halfPixel, mode, T_IDX>>(
            Simt::Dim3(THREAD_NUM), scaleH, scaleW, lenN, lenC, lenDesH, lenDesW, lenSrcH, lenSrcW, shiftH, mH, shiftW,
            mW, (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ volatile T2*)(outputGm_.GetPhyAddr()), blkStartOffset,
            blkProcessNum);
    }
}
} // namespace ResizeBilinearV2

#endif // CANN_RESIZE_BILINEAR_V2_SIMT_HW_H
