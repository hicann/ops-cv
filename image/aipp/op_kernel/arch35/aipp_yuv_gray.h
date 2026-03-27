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
 * \file aipp_yuv_gray.h
 * \brief aipp yuv to gray
 */

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H
#define AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippYuvGray : public AippBase<T, DataType> {
public:
    __aicore__ inline AippYuvGray(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuvGray<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

__aicore__ __attribute__((always_inline)) inline uint8_t Yuv2Gray(uint8_t y, const CscParam& cscParam)
{
    auto gTmp = static_cast<int16_t>(roundf(static_cast<float>(
        cscParam.cscMatrix00 * static_cast<int16_t>(y) * DIGIT_2 + 1) / CSC_MATRIX_SCALE));
    return CLIP3(gTmp, 0, MAX_UINT8);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void SimtComputeYuvGray(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;
    float padValue = tD.paddingParam.padValue;

    for (DataType idx = Simt::GetThreadIdx() + blockIdx * Simt::GetThreadNum(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum()) {
        CoordPack<DataType> coord;
        RgbPack<DataType> dstGrayIdx;
        ComputeCoordFromIndex(idx, outputSizeH, outputSizeW, coord);
        RgbComputeDstIdx(dstGrayIdx, coord, tD);
        bool isPadding = IsPixelInPadding(coord.hIdx, coord.wIdx, tD);

        if (isPadding) {
            AssignPadValue(outputGM[dstGrayIdx.r], padValue);
            AssignPadValue(outputGM[dstGrayIdx.g], padValue);
            AssignPadValue(outputGM[dstGrayIdx.b], padValue);
        } else {
            DataType srcYuvYIdx = coord.nIdx * tD.inputSizeH * tD.inputSizeW * DIGIT_3 / DIGIT_2 +
                (tD.cropParam.cropStartPosH + coord.hIdx - tD.paddingParam.topPaddingSize) * tD.inputSizeW +
                tD.cropParam.cropStartPosW + coord.wIdx - tD.paddingParam.leftPaddingSize;
            uint8_t grayValue = Yuv2Gray(yuvGM[srcYuvYIdx], tD.cscParam);
            DataConversion(outputGM[dstGrayIdx.r], grayValue, tD.dtcParam, CHANNEL_NUM_0);
            DataConversion(outputGM[dstGrayIdx.g], 0, tD.dtcParam, CHANNEL_NUM_1);
            DataConversion(outputGM[dstGrayIdx.b], 0, tD.dtcParam, CHANNEL_NUM_2);
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuvGray<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t batchSize = static_cast<uint64_t>(this->tilingData_.batchNum) *
        this->tilingData_.outputSizeH * this->tilingData_.outputSizeW;

    Simt::VF_CALL<Aipp_Kernel::SimtComputeYuvGray<T, DataType>>(Simt::Dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, batchSize);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H