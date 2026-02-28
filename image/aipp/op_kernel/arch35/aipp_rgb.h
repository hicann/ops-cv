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
 * \file aipp_rgb.h
 * \brief aipp rgb
 */

#ifndef AIPP_OP_KERNEL_ARCH35_RGB_H
#define AIPP_OP_KERNEL_ARCH35_RGB_H

#include "aipp_base.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippRgb : public AippBase<T, DataType> {
public:
    __aicore__ inline AippRgb(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = 1;
    uint16_t blockDimY_ = 1;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgb<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
    blockDimX_ = tilingData.channelNum;
    blockDimY_ = MAX_TGHREAD_NUM / blockDimX_;
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgb(
    __gm__ uint8_t* rgbGM, __gm__ T* outputGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint8_t outputFormat = tD.outputFormat;
    uint32_t channelNum = tD.channelNum;
    uint32_t inputSizeH = tD.inputSizeH;
    uint32_t inputSizeW = tD.inputSizeW;
    uint32_t cropSizeH = tD.cropParam.cropSizeH;
    uint32_t cropSizeW = tD.cropParam.cropSizeW;
    uint32_t cropStartPosH = tD.cropParam.cropStartPosH;
    uint32_t cropStartPosW = tD.cropParam.cropStartPosW;

    DataType threadIdxX = Simt::GetThreadIdx<0>();
    for (DataType idx = Simt::GetThreadIdx<1>() + blockIdx * Simt::GetThreadNum<1>(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum<1>()) {

        DataType nIdx = idx / (cropSizeH * cropSizeW);
        DataType newIdx = idx - nIdx * cropSizeH * cropSizeW;
        DataType hIdx = newIdx / cropSizeW;
        DataType wIdx = newIdx - hIdx * cropSizeW;

        DataType dstIdx = 0;
        if (outputFormat == NCHW_FORMAT_INDEX) {
            dstIdx = nIdx * (cropSizeH * cropSizeW * channelNum) +
                          threadIdxX * (cropSizeH * cropSizeW) + hIdx * cropSizeW + wIdx;
        } else {
            dstIdx = nIdx * (cropSizeH * cropSizeW * channelNum) +
                          hIdx * (cropSizeW * channelNum) + wIdx * channelNum + threadIdxX;
        }

        DataType srcIdx = nIdx * (inputSizeH * inputSizeW * channelNum) +
                          (cropStartPosH + hIdx) * inputSizeW * channelNum + (cropStartPosW + wIdx) * channelNum + threadIdxX;
        DataConversion(outputGM[dstIdx], rgbGM[srcIdx], tD.dtcParam, threadIdxX);
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgb<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    Simt::VF_CALL<Aipp_Kernel::SimtComputeRgb<T, DataType>>(Simt::Dim3(this->blockDimX_, this->blockDimY_), 
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, this->totalNum_);
}

}  // namespace Aipp_Kernel
#endif  // AIPP_OP_KERNEL_ARCH35_RGB_H
