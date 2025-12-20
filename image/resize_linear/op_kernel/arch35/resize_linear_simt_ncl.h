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
 * \file resize_linear_simt_ncl.h
 * \brief
 */

#ifndef CANN_RESIZE_LINEAR_SIMT_NCL_H
#define CANN_RESIZE_LINEAR_SIMT_NCL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ResizeLinear {
using namespace AscendC;

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
class ResizeLinearSimtNCL {
public:
    __aicore__ inline ResizeLinearSimtNCL(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeLinearTilingData *tilingData);
    __aicore__ inline void Process();

private:
    uint32_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
    const ResizeLinearTilingData *tilingData_;
};

template <typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline void ComputeMode0(
    float origWidth, T2 srcL1, T2 origBaseIdx, T2 yGmIdx, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    // 计算原图中坐标点
    T2 leftX = Simt::Floor(origWidth);
    if (leftX >= srcL1) {
        outputGm[yGmIdx] = inputGm[origBaseIdx + srcL1];
    } else {
        T2 rightX = leftX + 1;
        float pixelLeftTop = static_cast<float>(inputGm[origBaseIdx + leftX]);
        float deltaX = origWidth - static_cast<float>(leftX);
        float pixelRightTop = static_cast<float>(inputGm[origBaseIdx + rightX]);
        float value = (1.0f - deltaX) * pixelLeftTop + deltaX * pixelRightTop;
        // 根据周围2个点像素值进行双线性差值，计算出输出像素值
        outputGm[yGmIdx] = static_cast<T1>(value);
    }
}

template <typename T2, uint64_t halfPixel>
__aicore__ __attribute__((always_inline)) inline float ComputeOriL(T2 L, float scaleL)
{
    if constexpr (halfPixel == 1) {
        float origWidth = static_cast<float>((L + 0.5f) * scaleL) - 0.5f;
        if (origWidth < 0.0f) {
            origWidth = 0.0f;
        }
        return origWidth;
    } else {
        float origWidth = static_cast<float>(L * scaleL);
        return origWidth;
    }
}

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL,
    T2 lenDesL, T2 lenSrcL, float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    T2 srcL1 = lenSrcL - 1;
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
         idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 yGmIdx = blkStartOffset + idx;
        if constexpr (mode == 1) {
            // 纯搬运，输出完全等于输入，直接赋值
            outputGm[yGmIdx] = inputGm[yGmIdx];
            continue;
        }
        T2 NC = Simt::UintDiv(yGmIdx, mL, shiftL);
        if constexpr (mode == 2) {
            // 输入L等于1，输出broadcast即可
            outputGm[yGmIdx] = inputGm[NC];
            continue;
        }
        T2 origBaseIdx = NC * lenSrcL;
        if constexpr (mode == 3) {
            // dstL == 1, 取NC即可
            outputGm[yGmIdx] = inputGm[origBaseIdx];
            continue;
        }
        T2 L = yGmIdx - NC * lenDesL;
        float origWidth = ComputeOriL<T2, halfPixel>(L, scaleL);
        if constexpr (mode == 4) {
            // 逐点搬运，scale为整数，也就是权重为0时，直接取原始src坐标位置的点
            T2 leftX = Simt::Floor(origWidth);
            if (leftX > srcL1) {
                leftX = srcL1;
            }
            outputGm[yGmIdx] = inputGm[origBaseIdx + leftX];
            continue;
        }
        if constexpr (mode == 0) {
            ComputeMode0<T1, T2>(origWidth, srcL1, origBaseIdx, yGmIdx, inputGm, outputGm);
        }
    }
}

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
__simt_vf__ LAUNCH_BOUND(512) __aicore__ void calleeInt64(T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL,
    T2 lenDesL, T2 lenSrcL, float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtCompute<T1, T2, halfPixel, mode>(
        blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcL, scaleL, inputGm, outputGm);
}

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ void calleeInt32(T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL,
    T2 lenDesL, T2 lenSrcL, float scaleL, __gm__ T1 *inputGm, __gm__ T1 *outputGm)
{
    SimtCompute<T1, T2, halfPixel, mode>(
        blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcL, scaleL, inputGm, outputGm);
}

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
__aicore__ inline void ResizeLinearSimtNCL<T1, T2, halfPixel, mode>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeLinearTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1 *)x);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
}

template <typename T1, typename T2, uint64_t halfPixel, uint64_t mode>
__aicore__ inline void ResizeLinearSimtNCL<T1, T2, halfPixel, mode>::Process()
{
    if (blockIdx_ > tilingData_->realCoreNum - 1) {
        return;
    }
    T2 blkProcessNum = 0;
    T2 blkStartOffset = 0;
    if (blockIdx_ < tilingData_->splitBlockTailFactor) {
        blkProcessNum = tilingData_->blkProcessNum + 1;
        blkStartOffset = blockIdx_ * blkProcessNum;
    } else {
        blkProcessNum = tilingData_->blkProcessNum;
        blkStartOffset = tilingData_->splitBlockTailFactor * (tilingData_->blkProcessNum + 1) +
                         (blockIdx_ - tilingData_->splitBlockTailFactor) * blkProcessNum;
    }
    T2 mL = 0;
    T2 shiftL = 0;
    GetUintDivMagicAndShift(mL, shiftL, (T2)(tilingData_->lenDesL));

    T2 lenDesL = (T2)(tilingData_->lenDesL);
    T2 lenSrcL = (T2)(tilingData_->lenSrcL);
    float scaleL = tilingData_->scaleL;
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, T2, halfPixel, mode>>(Simt::Dim3(512),
            blkStartOffset,
            blkProcessNum,
            mL,
            shiftL,
            lenDesL,
            lenSrcL,
            scaleL,
            (__gm__ T1 *)(inputGm_.GetPhyAddr()),
            (__gm__ T1 *)(outputGm_.GetPhyAddr()));
    } else {
        Simt::VF_CALL<calleeInt32<T1, T2, halfPixel, mode>>(Simt::Dim3(1024),
            blkStartOffset,
            blkProcessNum,
            mL,
            shiftL,
            lenDesL,
            lenSrcL,
            scaleL,
            (__gm__ T1 *)(inputGm_.GetPhyAddr()),
            (__gm__ T1 *)(outputGm_.GetPhyAddr()));
    }
}
}  // namespace ResizeLinear
#endif  // CANN_RESIZE_LINEAR_SIMT_NCL_H
