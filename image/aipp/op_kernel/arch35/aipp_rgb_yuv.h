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
 * \file aipp_rgb_yuv.h
 * \brief aipp rgb to yuv kernel
 */

#ifndef AIPP_OP_KERNEL_ARCH35_RGB_YUV_H
#define AIPP_OP_KERNEL_ARCH35_RGB_YUV_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippRgbYuv : public AippBase<T, DataType> {
public:
    __aicore__ inline AippRgbYuv(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_TGHREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

__aicore__ __attribute__((always_inline)) inline void Rgb2Yuv(
    YuvPack<uint8_t>& dstYuv, uint8_t r, uint8_t g, uint8_t b, const CscParam& cscParam)
{
    auto yTmp = static_cast<int16_t>(roundf(static_cast<float>(cscParam.cscMatrix00 * r * 2 +
        cscParam.cscMatrix01 * g * 2 + cscParam.cscMatrix02 * b * 2 + 1) / CSC_MATRIX_SCALE));
    auto uTmp = static_cast<int16_t>(roundf(static_cast<float>(cscParam.cscMatrix10 * r * 2 +
        cscParam.cscMatrix11 * g * 2 + cscParam.cscMatrix12 * b * 2 + 1) / CSC_MATRIX_SCALE));
    auto vTmp = static_cast<int16_t>(roundf(static_cast<float>(cscParam.cscMatrix20 * r * 2 +
        cscParam.cscMatrix21 * g * 2 + cscParam.cscMatrix22 * b * 2 + 1) / CSC_MATRIX_SCALE));

    dstYuv.y = CLIP3(yTmp + cscParam.outBias0, 0, 255);
    dstYuv.u = CLIP3(uTmp + cscParam.outBias1, 0, 255);
    dstYuv.v = CLIP3(vTmp + cscParam.outBias2, 0, 255);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgb2Yuv(
    __gm__ uint8_t* rgbGM, __gm__ T* yuvGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint8_t outputFormat = tD.outputFormat;
    uint32_t inputSizeH = tD.inputSizeH;
    uint32_t inputSizeW = tD.inputSizeW;
    uint32_t cropSizeH = tD.cropParam.cropSizeH;
    uint32_t cropSizeW = tD.cropParam.cropSizeW;
    uint32_t cropStartPosH = tD.cropParam.cropStartPosH;
    uint32_t cropStartPosW = tD.cropParam.cropStartPosW;

    for (DataType idx = Simt::GetThreadIdx() + blockIdx * Simt::GetThreadNum(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum()) {
        DataType nIdx = idx / (cropSizeH * cropSizeW);
        DataType newIdx = idx - nIdx * cropSizeH * cropSizeW;
        DataType hIdx = newIdx / cropSizeW;
        DataType wIdx = newIdx - hIdx * cropSizeW;

        YuvPack<DataType> dstYuvIdx;
        if (outputFormat == NCHW_FORMAT_INDEX) {
            dstYuvIdx.y = nIdx * cropSizeH * cropSizeW * YUV_CHANNEL_NUM + hIdx * cropSizeW  + wIdx;
            dstYuvIdx.u = dstYuvIdx.y + cropSizeH * cropSizeW;
            dstYuvIdx.v = dstYuvIdx.u + cropSizeH * cropSizeW;
        } else {
            dstYuvIdx.y = nIdx * cropSizeH * cropSizeW * YUV_CHANNEL_NUM + hIdx * cropSizeW * YUV_CHANNEL_NUM + wIdx * YUV_CHANNEL_NUM;
            dstYuvIdx.u = dstYuvIdx.y + 1;
            dstYuvIdx.v = dstYuvIdx.u + 1;
        }

        RgbPack<DataType> srcRgbIdx;
        srcRgbIdx.r = nIdx * inputSizeH * inputSizeW * RGB_CHANNEL_NUM + (cropStartPosH + hIdx) * inputSizeW * RGB_CHANNEL_NUM +
                      (cropStartPosW + wIdx) * RGB_CHANNEL_NUM;
        srcRgbIdx.g = srcRgbIdx.r + 1;
        srcRgbIdx.b = srcRgbIdx.g + 1;

        YuvPack<uint8_t> dstYuv;
        Rgb2Yuv(dstYuv, rgbGM[srcRgbIdx.r], rgbGM[srcRgbIdx.g], rgbGM[srcRgbIdx.b], tD.cscParam);
        DataConversion(yuvGM[dstYuvIdx.y], dstYuv.y, tD.dtcParam, CHANNEL_NUM_0);
        DataConversion(yuvGM[dstYuvIdx.u], dstYuv.u, tD.dtcParam, CHANNEL_NUM_1);
        DataConversion(yuvGM[dstYuvIdx.v], dstYuv.v, tD.dtcParam, CHANNEL_NUM_2);
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    Simt::VF_CALL<Aipp_Kernel::SimtComputeRgb2Yuv<T, DataType>>(Simt::Dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, this->totalNum_);
}
} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_RGB_YUV_H