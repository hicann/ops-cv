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
 * \file aipp_gray.h
 * \brief aipp gray
 */

#ifndef AIPP_OP_KERNEL_ARCH35_GRAY_H
#define AIPP_OP_KERNEL_ARCH35_GRAY_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippRgbGray : public AippBase<T, DataType> {
public:
    __aicore__ inline AippRgbGray() {};
    __aicore__ inline void Init(const AippTilingData& tilingData,
        const tagAippDynamicParaHeader& tilingParamHeader,
        const __gm__ uint8_t* gmParams,
        uint8_t dynamicTilingKey);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgbGray<T, DataType>::Init(const AippTilingData& tilingData,
    const tagAippDynamicParaHeader& tilingParamHeader,
    const __gm__ uint8_t* gmParams,
    uint8_t dynamicTilingKey)
{
    this->BaseInit(tilingData, tilingParamHeader, gmParams, dynamicTilingKey);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void SimtComputeRgb2Gray(
    __gm__ uint8_t* rgbGM, __gm__ T* grayGM, AippTilingData tD,
    const __gm__ uint8_t* gmParams,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize, uint8_t dynamicTilingKey)
{
    float padValue = tD.paddingParam.padValue;
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;

    for (DataType idx = threadIdx.x + blockIdx * blockDim.x; idx < batchSize;
         idx += blockNum * blockDim.x) {
        CoordPack<DataType> coord;
        ComputeCoordFromIndex(idx, outputSizeH, outputSizeW, coord);
        if (dynamicTilingKey != 0) {
            UpdateDynamicBatchPara(coord, tD, gmParams);
        }

        RgbPack<DataType> dstGrayIdx;
        RgbComputeDstIdx(dstGrayIdx, coord, tD);

        bool isPadding = IsPixelInPadding(coord.hIdx, coord.wIdx, tD);

        if (isPadding) {
            AssignPadValue(grayGM[dstGrayIdx.r], padValue);
            AssignPadValue(grayGM[dstGrayIdx.g], padValue);
            AssignPadValue(grayGM[dstGrayIdx.b], padValue);
        } else {
            RgbPack<DataType> srcRgbIdx;
            RgbComputeSrcIdx(srcRgbIdx, coord, tD, (DataType)tD.srcChannelOffset);

            RgbPack<uint8_t> result;
            ApplyCscMatrix(result, rgbGM[srcRgbIdx.r], rgbGM[srcRgbIdx.g], rgbGM[srcRgbIdx.b], tD.cscParam);
            DataConversion(grayGM[dstGrayIdx.r], result.r, tD.dtcParam, CHANNEL_NUM_0);
            DataConversion(grayGM[dstGrayIdx.g], 0, tD.dtcParam, CHANNEL_NUM_1);
            DataConversion(grayGM[dstGrayIdx.b], 0, tD.dtcParam, CHANNEL_NUM_2);
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgbGray<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    asc_vf_call<Aipp_Kernel::SimtComputeRgb2Gray<T, DataType>>(dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->gmParams_,
         this->blockIdx_, this->blockNum_, this->totalNum_, this->dynamicTilingKey_);
}
} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_GRAY_H