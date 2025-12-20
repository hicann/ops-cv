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
 * \file resize_bicubic_v2_simt_nchw.h
 * \brief resize_bicubic_v2_simt_nchw
 */

#ifndef CANN_RESIZE_BICUBIC_V2_SIMT_NCHW_H
#define CANN_RESIZE_BICUBIC_V2_SIMT_NCHW_H

#include "kernel_operator.h"
#include "resize_bicubic_v2_simt_base.h"

namespace ResizeBicubicV2 {
using namespace AscendC;

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
class ResizeBicubicV2SimtNCHW {
public:
    __aicore__ inline ResizeBicubicV2SimtNCHW(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeBicubicV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    const ResizeBicubicV2TilingData *tilingData_;
    uint32_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(T_IDX blkStartOffset, T_IDX blkProcessNum, T_IDX mHw,
    T_IDX shiftHw, T_IDX mW, T_IDX shiftW, T_IDX mH, T_IDX shiftH, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDesH,
    T_IDX lenDesW, float scaleH, float scaleW, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    T_IDX lenSrcHw = lenSrcH * lenSrcW;
    T_IDX2 lenSrcH1 = lenSrcH - 1;
    T_IDX2 lenSrcW1 = lenSrcW - 1;
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) {
        T_IDX yGmIdx = blkStartOffset + idx;
        if constexpr (mode == 1) {
            // 纯搬运，输出完全等于输入，直接赋值
            outputGm[yGmIdx] = inputGm[yGmIdx];
            continue;
        }
        T_IDX NC = Simt::UintDiv(yGmIdx, mHw, shiftHw);
        if constexpr (mode == 2) {
            // 输入h 和w 等于1，输出broadcast即可
            outputGm[yGmIdx] = inputGm[NC];
            continue;
        } else if constexpr (mode == 3) {
            // dsth 和w==1,取NC即可
            outputGm[yGmIdx] = inputGm[NC * lenSrcHw];
            continue;
        }
        T_IDX H = 0;
        T_IDX W = 0;
        T_IDX tmpRes = Simt::UintDiv(yGmIdx, mW, shiftW);
        W = yGmIdx - tmpRes * lenDesW;

        T_IDX tmpRes1 = Simt::UintDiv(tmpRes, mH, shiftH);
        H = tmpRes - tmpRes1 * lenDesH;

        T_IDX origBaseIdx = NC * lenSrcHw;
        float origHeight = ComputeOri<T_IDX, halfPixel>(H, scaleH);
        float origWidth = ComputeOri<T_IDX, halfPixel>(W, scaleW);
        if constexpr (mode == 4) {
            T_IDX2 leftX = Simt::Floor(origWidth);
            T_IDX2 topY = Simt::Floor(origHeight);
            T_IDX2 newH = GetSrc<T_IDX2>(topY, lenSrcH1);
            T_IDX2 newW = GetSrc<T_IDX2>(leftX, lenSrcW1);
            outputGm[yGmIdx] = inputGm[origBaseIdx + newH * lenSrcW + newW];
            continue;
        }
        if constexpr (mode == 0) {
            ComputeNchwMode0<T1, T_IDX, T_IDX2>(
                origBaseIdx, yGmIdx, lenSrcW, origHeight, origWidth, lenSrcH1, lenSrcW1, inputGm, outputGm);
        }
    }
}

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__simt_vf__ LAUNCH_BOUND(256) __aicore__ void calleeInt64(T_IDX blkStartOffset, T_IDX blkProcessNum, T_IDX mHw,
    T_IDX shiftHw, T_IDX mW, T_IDX shiftW, T_IDX mH, T_IDX shiftH, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDesH,
    T_IDX lenDesW, float scaleH, float scaleW, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtCompute<T1, halfPixel, mode, T_IDX, T_IDX2>(blkStartOffset,
        blkProcessNum,
        mHw,
        shiftHw,
        mW,
        shiftW,
        mH,
        shiftH,
        lenSrcH,
        lenSrcW,
        lenDesH,
        lenDesW,
        scaleH,
        scaleW,
        inputGm,
        outputGm);
}

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__simt_vf__ LAUNCH_BOUND(512) __aicore__ void calleeInt32(T_IDX blkStartOffset, T_IDX blkProcessNum, T_IDX mHw,
    T_IDX shiftHw, T_IDX mW, T_IDX shiftW, T_IDX mH, T_IDX shiftH, T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDesH,
    T_IDX lenDesW, float scaleH, float scaleW, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtCompute<T1, halfPixel, mode, T_IDX, T_IDX2>(blkStartOffset,
        blkProcessNum,
        mHw,
        shiftHw,
        mW,
        shiftW,
        mH,
        shiftH,
        lenSrcH,
        lenSrcW,
        lenDesH,
        lenDesW,
        scaleH,
        scaleW,
        inputGm,
        outputGm);
}

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2SimtNCHW<T1, halfPixel, mode, T_IDX, T_IDX2>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeBicubicV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();

    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1 *)x);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
}

template <typename T1, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2SimtNCHW<T1, halfPixel, mode, T_IDX, T_IDX2>::Process()
{
    if (blockIdx_ > tilingData_->realCoreNum - 1) {
        return;
    }
    T_IDX blkProcessNum = 0;
    T_IDX blkStartOffset = 0;
    if (blockIdx_ < tilingData_->splitBlockTailFactor) {
        blkProcessNum = tilingData_->blkProcessNum + 1;
        blkStartOffset = blockIdx_ * blkProcessNum;
    } else {
        blkProcessNum = tilingData_->blkProcessNum;
        blkStartOffset = tilingData_->splitBlockTailFactor * (tilingData_->blkProcessNum + 1) +
                         (blockIdx_ - tilingData_->splitBlockTailFactor) * blkProcessNum;
    }
    T_IDX mHw = 0;
    T_IDX shiftHw = 0;
    T_IDX lenSrcH = tilingData_->lenSrcH;
    T_IDX lenSrcW = tilingData_->lenSrcW;
    T_IDX lenDesH = tilingData_->lenDesH;
    T_IDX lenDesW = tilingData_->lenDesW;
    float scaleH = tilingData_->scaleH;
    float scaleW = tilingData_->scaleW;
    T_IDX lenDesHw = lenDesH * lenDesW;
    GetUintDivMagicAndShift(mHw, shiftHw, lenDesHw);
    T_IDX mW = 0;
    T_IDX shiftW = 0;
    GetUintDivMagicAndShift(mW, shiftW, lenDesW);
    T_IDX mH = 0;
    T_IDX shiftH = 0;
    GetUintDivMagicAndShift(mH, shiftH, lenDesH);

    if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, halfPixel, mode, T_IDX, T_IDX2>>(Simt::Dim3(256),
            blkStartOffset,
            blkProcessNum,
            mHw,
            shiftHw,
            mW,
            shiftW,
            mH,
            shiftH,
            lenSrcH,
            lenSrcW,
            lenDesH,
            lenDesW,
            scaleH,
            scaleW,
            (__gm__ T1 *)(inputGm_.GetPhyAddr()),
            (__gm__ T1 *)(outputGm_.GetPhyAddr()));
    } else {
        Simt::VF_CALL<calleeInt32<T1, halfPixel, mode, T_IDX, T_IDX2>>(Simt::Dim3(512),
            blkStartOffset,
            blkProcessNum,
            mHw,
            shiftHw,
            mW,
            shiftW,
            mH,
            shiftH,
            lenSrcH,
            lenSrcW,
            lenDesH,
            lenDesW,
            scaleH,
            scaleW,
            (__gm__ T1 *)(inputGm_.GetPhyAddr()),
            (__gm__ T1 *)(outputGm_.GetPhyAddr()));
    }
}
}  // namespace ResizeBicubicV2

#endif  // CANN_RESIZE_BICUBIC_V2_SIMT_NCHW_H
