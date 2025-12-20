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
 * \file resize_bilinear_v2_grad_simt.h
 * \brief resize_bilinear_v2_grad_simt
 */

#ifndef CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_H
#define CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ResizeBilinearV2GradSimt {
using namespace AscendC;

constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t THREAD_NUM_MIDDLE = 512;

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
class ResizeBilinearV2GradSimt {
public:
    __aicore__ inline ResizeBilinearV2GradSimt(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, const ResizeBilinearV2GradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T2> outputGm_;
    int32_t blockIdx_;
    const ResizeBilinearV2GradTilingData *tiling_;
};

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ inline void ResizeBilinearV2GradSimt<T1, T2, halfPixel, T_IDX, format>::Init(
    GM_ADDR grads, GM_ADDR size, GM_ADDR y, const ResizeBilinearV2GradTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1 *)grads);
    outputGm_.SetGlobalBuffer((__gm__ T2 *)y);
}
//__aicore__ __attribute__((always_inline)) inline float
template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ __attribute__((always_inline)) inline void GetEachDimIdx(T_IDX gradsIdx, T_IDX &N, T_IDX &C, T_IDX &H,
    T_IDX &W, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW,
    T_IDX lenN, T_IDX lenC, T_IDX lenGradH, T_IDX lenGradW)
{
    T_IDX tmp = gradsIdx;
    if constexpr (format == FORMAT_NCHW) {
        T_IDX tmpRes = Simt::UintDiv(tmp, mW, shiftW);
        W = tmp - tmpRes * lenGradW;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mH, shiftH);
        H = tmp - tmpRes * lenGradH;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mC, shiftC);
        C = tmp - tmpRes * lenC;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mN, shiftN);
        N = tmp - tmpRes * lenN;
    } else {
        T_IDX tmpRes = Simt::UintDiv(tmp, mC, shiftC);
        C = tmp - tmpRes * lenC;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mW, shiftW);
        W = tmp - tmpRes * lenGradW;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mH, shiftH);
        H = tmp - tmpRes * lenGradH;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mN, shiftN);
        N = tmp - tmpRes * lenN;
    }
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ __attribute__((always_inline)) inline T_IDX CalOutputIdx(T_IDX gradsIdx, T_IDX N, T_IDX C, T_IDX h, T_IDX w,
    T_IDX lenC, T_IDX lenSrcW, T_IDX lenSrcH, T_IDX lenSrcWH, T_IDX lenSrcWC)
{
    if constexpr (format == FORMAT_NCHW) {
        return N * (lenC * lenSrcWH) + C * lenSrcWH + h * lenSrcW + w;
    } else {
        return N * (lenSrcH * lenSrcWC) + C + h * lenSrcWC + w * lenC;
    }
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(float scaleH, float scaleW, T_IDX lenN, T_IDX lenC,
    T_IDX lenGradH, T_IDX lenGradW, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1 *grads, __gm__ T2 *y, T_IDX blkStartOffset,
    T_IDX blkProcessNum)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX gradsIdx = blkStartOffset + idx;
        T_IDX N = 0, C = 0, H = 0, W = 0;
        GetEachDimIdx<T1, T2, halfPixel, T_IDX, format>(
            gradsIdx, N, C, H, W, shiftN, mN, shiftC, mC, shiftH, mH, shiftW, mW, lenN, lenC, lenGradH, lenGradW);
        T_IDX leftX = 0, rightX = 0, topY = 0, bottomY = 0;
        float origHeight = 0.0f, origWidth = 0.0f, deltaX = 0.0f, deltaX1 = 0.0f, deltaY = 0.0f, deltaY1 = 0.0f;
        T_IDX lenSrcWH = lenSrcW * lenSrcH;
        T_IDX lenSrcWC = lenSrcW * lenC;
        float offset = (halfPixel) ? HALF_PIXEL : 0.0f;
        origHeight = (static_cast<float>(H) + offset) * scaleH - offset;
        origWidth = (static_cast<float>(W) + offset) * scaleW - offset;

        origWidth = (origWidth < 0.0f) ? 0.0f : origWidth;
        leftX = Simt::Min(static_cast<T_IDX>(Simt::Floor(origWidth)), lenSrcW - 1);
        deltaX = Simt::Min(Simt::Max(origWidth - static_cast<float>(leftX), 0.0f), 1.0f);
        rightX = (leftX < lenSrcW - 1) ? leftX + 1 : leftX;
        deltaX1 = 1.0f - deltaX;

        origHeight = (origHeight < 0.0f) ? 0.0f : origHeight;
        topY = Simt::Min(static_cast<T_IDX>(Simt::Floor(origHeight)), lenSrcH - 1);
        deltaY = Simt::Min(Simt::Max(origHeight - static_cast<float>(topY), 0.0f), 1.0f);
        bottomY = (topY < lenSrcH - 1) ? topY + 1 : topY;
        deltaY1 = 1.0f - deltaY;

        T_IDX leftTopIdx = CalOutputIdx<T1, T2, halfPixel, T_IDX, format>(
            gradsIdx, N, C, topY, leftX, lenC, lenSrcW, lenSrcH, lenSrcWH, lenSrcWC);
        T_IDX rightTopIdx = CalOutputIdx<T1, T2, halfPixel, T_IDX, format>(
            gradsIdx, N, C, topY, rightX, lenC, lenSrcW, lenSrcH, lenSrcWH, lenSrcWC);
        T_IDX leftBotIdx = CalOutputIdx<T1, T2, halfPixel, T_IDX, format>(
            gradsIdx, N, C, bottomY, leftX, lenC, lenSrcW, lenSrcH, lenSrcWH, lenSrcWC);
        T_IDX rightBotIdx = CalOutputIdx<T1, T2, halfPixel, T_IDX, format>(
            gradsIdx, N, C, bottomY, rightX, lenC, lenSrcW, lenSrcH, lenSrcWH, lenSrcWC);

        float gradVal = static_cast<float>(grads[gradsIdx]);
        Simt::AtomicAdd(y + leftTopIdx, static_cast<T2>(deltaX1 * deltaY1 * gradVal));
        Simt::AtomicAdd(y + rightTopIdx, static_cast<T2>(deltaX * deltaY1 * gradVal));
        Simt::AtomicAdd(y + leftBotIdx, static_cast<T2>(deltaX1 * deltaY * gradVal));
        Simt::AtomicAdd(y + rightBotIdx, static_cast<T2>(deltaX * deltaY * gradVal));
    }
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__
    void calleeInt32(float scaleH, float scaleW, T_IDX lenN, T_IDX lenC, T_IDX lenGradH, T_IDX lenGradW, T_IDX lenSrcH,
        T_IDX lenSrcW, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW,
        __gm__ T1 *grads, __gm__ T2 *y, T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, T_IDX, format>(scaleH,
        scaleW,
        lenN,
        lenC,
        lenGradH,
        lenGradW,
        lenSrcH,
        lenSrcW,
        shiftN,
        mN,
        shiftC,
        mC,
        shiftH,
        mH,
        shiftW,
        mW,
        grads,
        y,
        blkStartOffset,
        blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_MIDDLE) __aicore__
    void calleeInt64(float scaleH, float scaleW, T_IDX lenN, T_IDX lenC, T_IDX lenGradH, T_IDX lenGradW, T_IDX lenSrcH,
        T_IDX lenSrcW, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW,
        __gm__ T1 *grads, __gm__ T2 *y, T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, T_IDX, format>(scaleH,
        scaleW,
        lenN,
        lenC,
        lenGradH,
        lenGradW,
        lenSrcH,
        lenSrcW,
        shiftN,
        mN,
        shiftC,
        mC,
        shiftH,
        mH,
        shiftW,
        mW,
        grads,
        y,
        blkStartOffset,
        blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ inline void ResizeBilinearV2GradSimt<T1, T2, halfPixel, T_IDX, format>::Process()
{
    T_IDX mH = 1;
    T_IDX mC = 1;
    T_IDX mW = 1;
    T_IDX mN = 1;
    T_IDX shiftN = 1;
    T_IDX shiftC = 1;
    T_IDX shiftH = 1;
    T_IDX shiftW = 1;
    GetUintDivMagicAndShift(mW, shiftW, static_cast<T_IDX>(tiling_->lenDesW));
    GetUintDivMagicAndShift(mH, shiftH, static_cast<T_IDX>(tiling_->lenDesH));
    GetUintDivMagicAndShift(mC, shiftC, static_cast<T_IDX>(tiling_->lenC));
    GetUintDivMagicAndShift(mN, shiftN, static_cast<T_IDX>(tiling_->lenN));

    T_IDX blkProcessNum = tiling_->splitBlockFactor;
    T_IDX blkStartOffset = blockIdx_ * tiling_->splitBlockFactor;
    T_IDX blkProcessNumY = tiling_->initYSplitBlockFactor;
    T_IDX blkStartOffsetY = blockIdx_ * tiling_->initYSplitBlockFactor;
    if (blockIdx_ < tiling_->splitBlockTailFactor) {
        blkProcessNum += 1;
        blkStartOffset += blockIdx_;
    } else {
        blkStartOffset += tiling_->splitBlockTailFactor;
    }
    if (blockIdx_ < tiling_->initYSplitBlockTailFactor) {
        blkProcessNumY += 1;
        blkStartOffsetY += blockIdx_;
    } else {
        blkStartOffsetY += tiling_->initYSplitBlockTailFactor;
    }
    const float scaleH = tiling_->scaleH;
    const float scaleW = tiling_->scaleW;
    const T_IDX lenSrcW = tiling_->lenSrcW;
    const T_IDX lenSrcH = tiling_->lenSrcH;
    const T_IDX lenC = tiling_->lenC;
    const T_IDX lenGradW = tiling_->lenDesW;
    const T_IDX lenGradH = tiling_->lenDesH;
    const T_IDX lenN = tiling_->lenN;
    const int32_t realCoreNum = tiling_->realCoreNum;
    const int32_t initYRealCoreNum = tiling_->initYRealCoreNum;
    if (blockIdx_ < initYRealCoreNum) {
        InitOutput<T2>(outputGm_[blkStartOffsetY], blkProcessNumY, static_cast<T2>(0.0f));
    }
    SyncAll();

    if (blockIdx_ < realCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64<T1, T2, halfPixel, T_IDX, format>>(Simt::Dim3(THREAD_NUM_MIDDLE),
                scaleH,
                scaleW,
                lenN,
                lenC,
                lenGradH,
                lenGradW,
                lenSrcH,
                lenSrcW,
                shiftN,
                mN,
                shiftC,
                mC,
                shiftH,
                mH,
                shiftW,
                mW,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()),
                (__gm__ T2 *)(outputGm_.GetPhyAddr()),
                blkStartOffset,
                blkProcessNum);
        } else {
            Simt::VF_CALL<calleeInt32<T1, T2, halfPixel, T_IDX, format>>(Simt::Dim3(THREAD_NUM),
                scaleH,
                scaleW,
                lenN,
                lenC,
                lenGradH,
                lenGradW,
                lenSrcH,
                lenSrcW,
                shiftN,
                mN,
                shiftC,
                mC,
                shiftH,
                mH,
                shiftW,
                mW,
                (__gm__ T1 *)(inputGm_.GetPhyAddr()),
                (__gm__ T2 *)(outputGm_.GetPhyAddr()),
                blkStartOffset,
                blkProcessNum);
        }
    }
}
}  // namespace ResizeBilinearV2GradSimt

#endif  // CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_H
