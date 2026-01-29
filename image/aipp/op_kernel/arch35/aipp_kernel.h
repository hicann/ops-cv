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
 * \file aipp_kernel.h
 * \brief aipp kernel
 */

#ifndef AIPP_OP_KERNEL_ARCH35_KERNEL_H
#define AIPP_OP_KERNEL_ARCH35_KERNEL_H

#include "aipp_base.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippRgb : public AippBase<T, DataType> {
public:
    __aicore__ inline AippRgb(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    uint16_t blockDimX_ = 1;
    uint16_t blockDimY_ = 1;
    uint32_t mBlockIdx_;
    const AippTilingData* mTD_;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgb<T, DataType>::Init(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    this->BaseInit(x, y, tilingData);
    blockDimX_ = this->channelNum_;
    blockDimY_ = MAX_TGHREAD_NUM / blockDimX_;
    mTD_ = &tilingData;
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgbNHWC(
    __gm__ uint8_t* inputGM, __gm__ T* outputGM, GM_ADDR tiling, uint32_t mBlockIdx_, uint32_t blockNum,
    DataType batchSize)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(AippTilingData, tD, tiling);
    int32_t threadIdxX = Simt::GetThreadIdx<0>();

    for (DataType idx = Simt::GetThreadIdx<1>() + mBlockIdx_ * Simt::GetThreadNum<1>(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum<1>()) {
        DataType nIdx = idx / (tD->cropSizeH * tD->cropSizeW);
        DataType newIdx = idx - nIdx * tD->cropSizeH * tD->cropSizeW;
        DataType hIdx = newIdx / tD->cropSizeW;
        DataType wIdx = newIdx - hIdx * tD->cropSizeW;

        DataType dstIdx = nIdx * (tD->cropSizeH * tD->cropSizeW * tD->channelNum) +
                          hIdx * (tD->cropSizeW * tD->channelNum) + wIdx * tD->channelNum + threadIdxX;

        DataType srcIdx = nIdx * (tD->inputSizeH * tD->inputSizeW * tD->channelNum) +
                          (tD->cropStartPosH + hIdx) * (tD->inputSizeW * tD->channelNum) +
                          (tD->cropStartPosW + wIdx) * tD->channelNum + threadIdxX;

        if constexpr (sizeof(T) == DIGIT_TWO) {
            uint8_t src = inputGM[srcIdx];
            T dstFp;
            if (threadIdxX == 0) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn0) - static_cast<T>(tD->dtcPixelMinChn0)) *
                        static_cast<T>(tD->dtcPixelVarReciChn0);
            } else if (threadIdxX == 1) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn1) - static_cast<T>(tD->dtcPixelMinChn1)) *
                        static_cast<T>(tD->dtcPixelVarReciChn1);
            } else if (threadIdxX == 2) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn2) - static_cast<T>(tD->dtcPixelMinChn2)) *
                        static_cast<T>(tD->dtcPixelVarReciChn2);
            } else {
                dstFp = 0;
            }
            outputGM[dstIdx] = dstFp;
        } else {
            outputGM[dstIdx] = inputGM[srcIdx];
        }
    }
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgbNCHW(
    __gm__ uint8_t* inputGM, __gm__ T* outputGM, GM_ADDR tiling, uint32_t mBlockIdx_, uint32_t blockNum,
    DataType batchSize)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(AippTilingData, tD, tiling);
    int32_t threadIdxX = Simt::GetThreadIdx<0>();

    for (DataType idx = Simt::GetThreadIdx<1>() + mBlockIdx_ * Simt::GetThreadNum<1>(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum<1>()) {
        DataType nIdx = idx / (tD->cropSizeH * tD->cropSizeW);
        DataType newIdx = idx - nIdx * tD->cropSizeH * tD->cropSizeW;
        DataType hIdx = newIdx / tD->cropSizeW;
        DataType wIdx = newIdx - hIdx * tD->cropSizeW;

        DataType dstIdx = nIdx * (tD->cropSizeH * tD->cropSizeW * tD->channelNum) +
                          threadIdxX * (tD->cropSizeH * tD->cropSizeW) + hIdx * tD->cropSizeW + wIdx;

        DataType srcIdx = nIdx * (tD->inputSizeH * tD->inputSizeW * tD->channelNum) +
                          threadIdxX * (tD->inputSizeH * tD->inputSizeW) + (tD->cropStartPosH + hIdx) * tD->inputSizeW +
                          (tD->cropStartPosW + wIdx);

        if constexpr (sizeof(T) == DIGIT_TWO) {
            uint8_t src = inputGM[srcIdx];
            T dstFp;
            if (threadIdxX == 0) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn0) - static_cast<T>(tD->dtcPixelMinChn0)) *
                        static_cast<T>(tD->dtcPixelVarReciChn0);
            } else if (threadIdxX == 1) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn1) - static_cast<T>(tD->dtcPixelMinChn1)) *
                        static_cast<T>(tD->dtcPixelVarReciChn1);
            } else if (threadIdxX == 2) {
                dstFp = (static_cast<T>(src - tD->dtcPixelMeanChn2) - static_cast<T>(tD->dtcPixelMinChn2)) *
                        static_cast<T>(tD->dtcPixelVarReciChn2);
            } else {
                dstFp = 0;
            }
            outputGM[dstIdx] = dstFp;
        } else {
            outputGM[dstIdx] = inputGM[srcIdx];
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgb<T, DataType>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum();
    if (this->inputFormat_ == NHWC_FORMAT_INDEX) {
        Simt::VF_CALL<SimtComputeRgbNHWC<T, DataType>>(
            Simt::Dim3(this->blockDimX_, this->blockDimY_), (__gm__ uint8_t*)(this->xGm_.GetPhyAddr()),
            (__gm__ T*)(this->yGm_.GetPhyAddr()), tiling, mBlockIdx_, blockNum, this->totalNum_);
    } else {
        Simt::VF_CALL<SimtComputeRgbNCHW<T, DataType>>(
            Simt::Dim3(this->blockDimX_, this->blockDimY_), (__gm__ uint8_t*)(this->xGm_.GetPhyAddr()),
            (__gm__ T*)(this->yGm_.GetPhyAddr()), tiling, mBlockIdx_, blockNum, this->totalNum_);
    }
}

}  // namespace Aipp
#endif  // AIPP_OP_KERNEL_ARCH35_KERNEL_H
