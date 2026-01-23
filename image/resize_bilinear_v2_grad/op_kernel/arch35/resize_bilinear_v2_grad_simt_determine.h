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
 * \file resize_bilinear_v2_grad_simt_determine.h
 * \brief resize_bilinear_v2_grad_simt_determine
 */

#ifndef CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_DETERMINE_H
#define CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_DETERMINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_bilinear_v2_grad_base.h"

namespace ResizeBilinearV2Grad {
using namespace AscendC;

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
class ResizeBilinearV2GradSimtDetermine {
public:
    __aicore__ inline ResizeBilinearV2GradSimtDetermine(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, const ResizeBilinearV2GradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T1> gradsGm_;
    GlobalTensor<T2> originalImageGm_;
    GlobalTensor<T2> yGm_;
    int32_t blockIdx_;
    const ResizeBilinearV2GradTilingData* tilingData_;
};

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ __attribute__((always_inline)) inline void
ResizeBilinearV2GradSimtDetermine<T1, T2, halfPixel, T_IDX, format>::Init(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, const ResizeBilinearV2GradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    gradsGm_.SetGlobalBuffer((__gm__ T1*)grads);
    yGm_.SetGlobalBuffer((__gm__ T2*)y);
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(
    float scaleW, float scaleH, float inverseScaleW, float inverseScaleH, T_IDX resizedHeight, T_IDX resizedWidth,
    T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX coreNum, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* grads, __gm__ T2* y, int32_t blockId,
    T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    if (blockId >= coreNum) {
        return;
    }
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yGmIdx = blkStartOffset + idx;
        T_IDX tmp = yGmIdx;

        T_IDX N = 0, C = 0, H = 0, W = 0;

        // 快速整除计算
        if constexpr (format == FORMAT_NCHW) {
            QuickDivForSimtComputeDetermine(
                tmp, mW, shiftW, W, lenSrcW, mH, shiftH, H, lenSrcH, mC, shiftC, C, lenC, mN, shiftN, N, lenN);
        } else {
            QuickDivForSimtComputeDetermine(
                tmp, mC, shiftC, C, lenC, mW, shiftW, W, lenSrcW, mH, shiftH, H, lenSrcH, mN, shiftN, N, lenN);
        }

        float offset = 0.0f;
        if constexpr (halfPixel) {
            offset = HALF_PIXEL;
        }

        T_IDX inYStart = static_cast<T_IDX>(Simt::Max(
            static_cast<int64_t>(0),
            static_cast<int64_t>(Simt::Ceil((static_cast<float>(H) - 1.0f + offset) * inverseScaleH - offset))));
        const float outYStart = (static_cast<float>(inYStart) + offset) * scaleH - offset;
        T_IDX inXStart = static_cast<T_IDX>(Simt::Max(
            static_cast<int64_t>(0),
            static_cast<int64_t>(Simt::Ceil((static_cast<float>(W) - 1.0f + offset) * inverseScaleW - offset))));
        const float outXStart = (static_cast<float>(inXStart) + offset) * scaleW - offset;
        float acc = 0, outY = outYStart;
        T_IDX inY = inYStart;
        while (outY < (static_cast<float>(H) + 1.0f) && inY < resizedHeight) {
            float outX = outXStart;
            T_IDX inX = inXStart;
            while (outX < (static_cast<float>(W) + 1.0f) && inX < resizedWidth) {
                T_IDX inIdx = 0;
                if constexpr (format == FORMAT_NCHW) {
                    inIdx = ((N * lenC + C) * resizedHeight + inY) * resizedWidth + inX;
                } else {
                    inIdx = ((N * resizedHeight + inY) * resizedWidth + inX) * lenC + C;
                }

                // Clamping to zero is necessary because outX and outY can be negative
                // due to half-pixel adjustments to outYStart and outXStart.
                // Clamping to height/width is necessary when upscaling.
                float outYClamped = Simt::Max(0.0f, Simt::Min(outY, static_cast<float>(lenSrcH) - 1.0f));
                float outXClamped = Simt::Max(0.0f, Simt::Min(outX, static_cast<float>(lenSrcW) - 1.0f));
                float yLerp = (1.0f - Simt::Abs(outYClamped - static_cast<float>(H)));
                float xLerp = (1.0f - Simt::Abs(outXClamped - static_cast<float>(W)));
                float gradVal = static_cast<float>(grads[inIdx]);
                acc += gradVal * yLerp * xLerp;
                outX += scaleW;
                inX++;
            }
            outY += scaleH;
            inY++;
        }
        y[yGmIdx] = static_cast<T2>(acc);
    }
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void calleeInt32Determine(
    float scaleW, float scaleH, float inverseScaleW, float inverseScaleH, T_IDX resizedHeight, T_IDX resizedWidth,
    T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX coreNum, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* grads, __gm__ T2* y, int32_t blockId,
    T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, T_IDX, format>(
        scaleW, scaleH, inverseScaleW, inverseScaleH, resizedHeight, resizedWidth, lenN, lenC, lenSrcH, lenSrcW,
        coreNum, shiftN, mN, shiftC, mC, shiftH, mH, shiftW, mW, grads, y, blockId, blkStartOffset, blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_MIDDLE) __aicore__ void calleeInt64Determine(
    float scaleW, float scaleH, float inverseScaleW, float inverseScaleH, T_IDX resizedHeight, T_IDX resizedWidth,
    T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX coreNum, T_IDX shiftN, T_IDX mN, T_IDX shiftC, T_IDX mC,
    T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, __gm__ T1* grads, __gm__ T2* y, int32_t blockId,
    T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T1, T2, halfPixel, T_IDX, format>(
        scaleW, scaleH, inverseScaleW, inverseScaleH, resizedHeight, resizedWidth, lenN, lenC, lenSrcH, lenSrcW,
        coreNum, shiftN, mN, shiftC, mC, shiftH, mH, shiftW, mW, grads, y, blockId, blkStartOffset, blkProcessNum);
}

template <typename T1, typename T2, bool halfPixel, typename T_IDX, int32_t format>
__aicore__ inline void ResizeBilinearV2GradSimtDetermine<T1, T2, halfPixel, T_IDX, format>::Process()
{
    T_IDX shiftN_ = 1;
    T_IDX mN = 1;
    T_IDX shiftC_ = 1;
    T_IDX mC = 1;
    T_IDX shiftH_ = 1;
    T_IDX mH = 1;
    T_IDX shiftW_ = 1;
    T_IDX mW = 1;

    GetUintDivMagicAndShift(mW, shiftW_, static_cast<T_IDX>(tilingData_->lenSrcW));
    GetUintDivMagicAndShift(mH, shiftH_, static_cast<T_IDX>(tilingData_->lenSrcH));
    GetUintDivMagicAndShift(mC, shiftC_, static_cast<T_IDX>(tilingData_->lenC));
    GetUintDivMagicAndShift(mN, shiftN_, static_cast<T_IDX>(tilingData_->lenN));

    T_IDX blkProcessNum = tilingData_->splitBlockFactor;
    T_IDX blkStartOffset = blockIdx_ * tilingData_->splitBlockFactor;
    if (blockIdx_ < tilingData_->splitBlockTailFactor) {
        blkProcessNum += 1;
        blkStartOffset += blockIdx_;
    } else {
        blkStartOffset += tilingData_->splitBlockTailFactor;
    }

    if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64Determine<T1, T2, halfPixel, T_IDX, format>>(
            Simt::Dim3{THREAD_NUM_MIDDLE, 1, 1}, tilingData_->scaleW, tilingData_->scaleH, tilingData_->inverseScaleW,
            tilingData_->inverseScaleH, tilingData_->lenDesH, tilingData_->lenDesW, tilingData_->lenN,
            tilingData_->lenC, tilingData_->lenSrcH, tilingData_->lenSrcW, tilingData_->realCoreNum, shiftN_, mN,
            shiftC_, mC, shiftH_, mH, shiftW_, mW, (__gm__ T1*)(gradsGm_.GetPhyAddr()), (__gm__ T2*)(yGm_.GetPhyAddr()),
            blockIdx_, blkStartOffset, blkProcessNum);
    } else {
        Simt::VF_CALL<calleeInt32Determine<T1, T2, halfPixel, T_IDX, format>>(
            Simt::Dim3{THREAD_NUM, 1, 1}, tilingData_->scaleW, tilingData_->scaleH, tilingData_->inverseScaleW,
            tilingData_->inverseScaleH, tilingData_->lenDesH, tilingData_->lenDesW, tilingData_->lenN,
            tilingData_->lenC, tilingData_->lenSrcH, tilingData_->lenSrcW, tilingData_->realCoreNum, shiftN_, mN,
            shiftC_, mC, shiftH_, mH, shiftW_, mW, (__gm__ T1*)(gradsGm_.GetPhyAddr()), (__gm__ T2*)(yGm_.GetPhyAddr()),
            blockIdx_, blkStartOffset, blkProcessNum);
    }
}
} // namespace ResizeBilinearV2Grad

#endif // CANN_RESIZE_BILINEAR_V2_GRAD_SIMT_DETERMINE_H