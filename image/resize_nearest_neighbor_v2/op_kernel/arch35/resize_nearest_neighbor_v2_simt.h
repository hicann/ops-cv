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
 * \file resize_nearest_neighbor_v2_simt.h
 * \brief resize_nearest_neighbor_v2_simt
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_SIMT_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t THREAD_NUM_MIDDLE = 512;
constexpr int ALL_COPY_MODE = 4;
constexpr int INPUT_ONE_MODE = 5;

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
class ResizeNearestNeighborV2Simt {
public:
    __aicore__ inline ResizeNearestNeighborV2Simt(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeNearestNeighborV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    int32_t blockIdx_;
    const ResizeNearestNeighborV2TilingData *tiling_;
};

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ inline void ResizeNearestNeighborV2Simt<T, T_IDX, format, mode, align_corner, half_pixel>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeNearestNeighborV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T *)x);
    outputGm_.SetGlobalBuffer((__gm__ T *)y);
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ __attribute__((always_inline)) inline void GetEachDimIdx(T_IDX yGmIdx, T_IDX &N, T_IDX &C, T_IDX &H,
    T_IDX &W, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, T_IDX lenC, T_IDX lenDesH,
    T_IDX lenDesW)
{
    T_IDX tmp = yGmIdx;
    if constexpr ((format == FORMAT_NCHW) || (format == FORMAT_ND)) {
        T_IDX tmpRes = Simt::UintDiv(tmp, mW, shiftW);
        W = tmp - tmpRes * lenDesW;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mH, shiftH);
        H = tmp - tmpRes * lenDesH;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mC, shiftC);
        C = tmp - tmpRes * lenC;
        N = tmpRes;
    } else {
        T_IDX tmpRes = Simt::UintDiv(tmp, mC, shiftC);
        C = tmp - tmpRes * lenC;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mW, shiftW);
        W = tmp - tmpRes * lenDesW;
        tmp = tmpRes;
        tmpRes = Simt::UintDiv(tmp, mH, shiftH);
        H = tmp - tmpRes * lenDesH;
        N = tmpRes;
    }
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ __attribute__((always_inline)) inline T_IDX CalOutputIdx(
    T_IDX N, T_IDX C, T_IDX h, T_IDX w, T_IDX lenC, T_IDX lenSrcW, T_IDX lenSrcH, T_IDX lenSrcWH, T_IDX lenSrcWC)
{
    if constexpr ((format == FORMAT_NCHW) || (format == FORMAT_ND)) {
        return N * (lenC * lenSrcWH) + C * lenSrcWH + h * lenSrcW + w;
    } else {
        return N * (lenSrcH * lenSrcWC) + C + h * lenSrcWC + w * lenC;
    }
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ __attribute__((always_inline)) inline void CalSrcIdx(
    T_IDX h, T_IDX w, T_IDX lenSrcW, T_IDX lenSrcH, T_IDX &origH, T_IDX &origW, float scaleH, float scaleW)
{
    T_IDX origHeight = 0;
    T_IDX origWidth = 0;
    if constexpr (align_corner && !half_pixel) {
        origHeight = Simt::Round(static_cast<float>(h * scaleH));
        origWidth = Simt::Round(static_cast<float>(w * scaleW));
    } else if constexpr (!align_corner && half_pixel) {
        origHeight = Simt::Floor(static_cast<float>((h + HALF_PIXEL) * scaleH));
        origWidth = Simt::Floor(static_cast<float>((w + HALF_PIXEL) * scaleW));
    } else if constexpr (!align_corner && !half_pixel) {
        origHeight = Simt::Floor(static_cast<float>(h * scaleH));
        origWidth = Simt::Floor(static_cast<float>(w * scaleW));
    }

    origH = Simt::Min(origHeight, lenSrcH - 1);
    origW = Simt::Min(origWidth, lenSrcW - 1);
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(float scaleH, float scaleW, T_IDX lenC, T_IDX lenDesH,
    T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW,
    T_IDX lenSrcWH, T_IDX lenSrcWC, __gm__ T *inputGm, __gm__ T *outputGm, T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yGmIdx = blkStartOffset + idx;
        if constexpr (mode == ALL_COPY_MODE) {
            outputGm[yGmIdx] = inputGm[yGmIdx];
            continue;
        }

        T_IDX N = 0, C = 0, H = 0, W = 0;
        GetEachDimIdx<T, T_IDX, format, mode, align_corner, half_pixel>(
            yGmIdx, N, C, H, W, shiftC, mC, shiftH, mH, shiftW, mW, lenC, lenDesH, lenDesW);
        if constexpr (mode == INPUT_ONE_MODE) {
            outputGm[yGmIdx] = inputGm[CalOutputIdx<T, T_IDX, format, mode, align_corner, half_pixel>(
                N, C, 0, 0, lenC, 1, 1, 1, lenC)];
            continue;
        }

        T_IDX origH = 0, origW = 0;
        CalSrcIdx<T, T_IDX, format, mode, align_corner, half_pixel>(
            H, W, lenSrcW, lenSrcH, origH, origW, scaleH, scaleW);
        T_IDX outputGmIdx = CalOutputIdx<T, T_IDX, format, mode, align_corner, half_pixel>(
            N, C, origH, origW, lenC, lenSrcW, lenSrcH, lenSrcWH, lenSrcWC);
        outputGm[yGmIdx] = inputGm[outputGmIdx];
    }
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void calleeInt32(float scaleH, float scaleW, T_IDX lenC, T_IDX lenDesH,
    T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW,
    T_IDX lenSrcWH, T_IDX lenSrcWC, __gm__ T *inputGm, __gm__ T *outputGm, T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T, T_IDX, format, mode, align_corner, half_pixel>(scaleH,
        scaleW,
        lenC,
        lenDesH,
        lenDesW,
        lenSrcH,
        lenSrcW,
        shiftC,
        mC,
        shiftH,
        mH,
        shiftW,
        mW,
        lenSrcWH,
        lenSrcWC,
        inputGm,
        outputGm,
        blkStartOffset,
        blkProcessNum);
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_MIDDLE) __aicore__
    void calleeInt64(float scaleH, float scaleW, T_IDX lenC, T_IDX lenDesH, T_IDX lenDesW, T_IDX lenSrcH, T_IDX lenSrcW,
        T_IDX shiftC, T_IDX mC, T_IDX shiftH, T_IDX mH, T_IDX shiftW, T_IDX mW, T_IDX lenSrcWH, T_IDX lenSrcWC,
        __gm__ T *inputGm, __gm__ T *outputGm, T_IDX blkStartOffset, T_IDX blkProcessNum)
{
    SimtCompute<T, T_IDX, format, mode, align_corner, half_pixel>(scaleH,
        scaleW,
        lenC,
        lenDesH,
        lenDesW,
        lenSrcH,
        lenSrcW,
        shiftC,
        mC,
        shiftH,
        mH,
        shiftW,
        mW,
        lenSrcWH,
        lenSrcWC,
        inputGm,
        outputGm,
        blkStartOffset,
        blkProcessNum);
}

template <typename T, typename T_IDX, int format, int mode, bool align_corner, bool half_pixel>
__aicore__ inline void ResizeNearestNeighborV2Simt<T, T_IDX, format, mode, align_corner, half_pixel>::Process()
{
    T_IDX mH = 1, mC = 1, mW = 1;
    T_IDX shiftC = 1, shiftH = 1, shiftW = 1;
    GetUintDivMagicAndShift(mW, shiftW, static_cast<T_IDX>(tiling_->lenDesW));
    GetUintDivMagicAndShift(mH, shiftH, static_cast<T_IDX>(tiling_->lenDesH));
    GetUintDivMagicAndShift(mC, shiftC, static_cast<T_IDX>(tiling_->lenC));

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
    const int32_t realCoreNum = tiling_->realCoreNum;
    const T_IDX lenSrcWH = lenSrcW * lenSrcH;
    const T_IDX lenSrcWC = lenSrcW * lenC;

    if (blockIdx_ < realCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64<T, T_IDX, format, mode, align_corner, half_pixel>>(Simt::Dim3(THREAD_NUM_MIDDLE),
                scaleH,
                scaleW,
                lenC,
                lenDesH,
                lenDesW,
                lenSrcH,
                lenSrcW,
                shiftC,
                mC,
                shiftH,
                mH,
                shiftW,
                mW,
                lenSrcWH,
                lenSrcWC,
                (__gm__ T *)(inputGm_.GetPhyAddr()),
                (__gm__ T *)(outputGm_.GetPhyAddr()),
                blkStartOffset,
                blkProcessNum);
        } else {
            Simt::VF_CALL<calleeInt32<T, T_IDX, format, mode, align_corner, half_pixel>>(Simt::Dim3(THREAD_NUM),
                scaleH,
                scaleW,
                lenC,
                lenDesH,
                lenDesW,
                lenSrcH,
                lenSrcW,
                shiftC,
                mC,
                shiftH,
                mH,
                shiftW,
                mW,
                lenSrcWH,
                lenSrcWC,
                (__gm__ T *)(inputGm_.GetPhyAddr()),
                (__gm__ T *)(outputGm_.GetPhyAddr()),
                blkStartOffset,
                blkProcessNum);
        }
    }
}
}  // namespace ResizeNearestNeighborV2

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_SIMT_H
