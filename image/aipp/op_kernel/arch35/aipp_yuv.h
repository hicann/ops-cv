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
 * \file aipp_yuv.h
 * \brief aipp yuv
 */

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_H
#define AIPP_OP_KERNEL_ARCH35_YUV_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippYuv : public AippBase<T, DataType> {
public:
    __aicore__ inline AippYuv(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_TGHREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void ComputeDstIdx(
    RgbPack<DataType>* dstRgbIdx, const CoordPack<DataType>& coord, uint32_t cropSizeH, 
    uint64_t cropSizeW, uint8_t outputFormat)
{
    if (outputFormat == NCHW_FORMAT_INDEX) {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (cropSizeH * cropSizeW * RGB_CHANNEL_NUM) + 
                                DIGIT_TWO * coord.hIdx * cropSizeW  + DIGIT_TWO * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + cropSizeH * cropSizeW;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + cropSizeH * cropSizeW;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + 1;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (cropSizeH * cropSizeW * RGB_CHANNEL_NUM) +
                                    (DIGIT_TWO * coord.hIdx + 1) * cropSizeW + DIGIT_TWO * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + cropSizeH * cropSizeW;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + cropSizeH * cropSizeW;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + 1;
    } else {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (cropSizeH * cropSizeW * RGB_CHANNEL_NUM) + 
                                DIGIT_TWO * coord.hIdx * cropSizeW * RGB_CHANNEL_NUM +
                                DIGIT_TWO * coord.wIdx * RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + RGB_CHANNEL_NUM;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (cropSizeH * cropSizeW * RGB_CHANNEL_NUM) +
                                    (DIGIT_TWO * coord.hIdx + 1) * cropSizeW * RGB_CHANNEL_NUM +
                                    DIGIT_TWO * coord.wIdx * RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + RGB_CHANNEL_NUM;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + RGB_CHANNEL_NUM;
    }
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void ComputeSrcIdx(
    DataType* srcYIdx, const CoordPack<DataType>& coord, uint32_t cropStartPosH,
    uint32_t cropStartPosW, uint32_t inputSizeW, uint64_t inputSizeH)
{
    srcYIdx[YUV_DEAL_NUM_0] = coord.nIdx * inputSizeH * inputSizeW * 3 / 2 + 
                        (cropStartPosH + DIGIT_TWO * coord.hIdx) * inputSizeW +
                            (cropStartPosW + DIGIT_TWO * coord.wIdx);
    srcYIdx[YUV_DEAL_NUM_1] = srcYIdx[YUV_DEAL_NUM_0] + 1;
    srcYIdx[YUV_DEAL_NUM_2] = coord.nIdx * inputSizeH * inputSizeW * 3 / 2 +
                            (cropStartPosH + DIGIT_TWO * coord.hIdx + 1) * inputSizeW +
                            (cropStartPosW + DIGIT_TWO * coord.wIdx);
    srcYIdx[YUV_DEAL_NUM_3] = srcYIdx[YUV_DEAL_NUM_2] + 1;
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void Yuv2Rgb(
    RgbPack<T>& dstRgb, uint8_t y, uint8_t u, uint8_t v, const CscParam& cscParam)
{
    auto rTmp = static_cast<int16_t>(y) - cscParam.inBias0;
    auto gTmp = static_cast<int16_t>(u) - cscParam.inBias1;
    auto bTmp = static_cast<int16_t>(v) - cscParam.inBias2;

    dstRgb.r = CLIP3(roundf(static_cast<float>(cscParam.cscMatrix00 * rTmp * 2 +
        cscParam.cscMatrix01 * gTmp * 2 + cscParam.cscMatrix02 * bTmp * 2 + 1) / CSC_MATRIX_SCALE), 0, 255);
    dstRgb.g = CLIP3(roundf(static_cast<float>(cscParam.cscMatrix10 * rTmp * 2 +
        cscParam.cscMatrix11 * gTmp * 2 + cscParam.cscMatrix12 * bTmp * 2 + 1) / CSC_MATRIX_SCALE), 0, 255);
    dstRgb.b = CLIP3(roundf(static_cast<float>(cscParam.cscMatrix20 * rTmp * 2 +
        cscParam.cscMatrix21 * gTmp * 2 + cscParam.cscMatrix22 * bTmp * 2 + 1) / CSC_MATRIX_SCALE), 0, 255);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeYuv(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const AippTilingData tD,
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
        CoordPack<DataType> coord;
        coord.nIdx = idx / ((cropSizeH >> 1) * (cropSizeW >> 1)) ;
        DataType newIdx = idx - coord.nIdx *  ((cropSizeH >> 1) * (cropSizeW >> 1));
        coord.hIdx = newIdx / (cropSizeW >> 1);
        coord.wIdx = newIdx - coord.hIdx * (cropSizeW >> 1);

        RgbPack<DataType> dstRgbIdx[YUV_PER_DEAL_NUM];
        ComputeDstIdx(dstRgbIdx, coord, cropSizeH, cropSizeW, outputFormat);

        // YUV420SP_U8 format: 1 UV value shared for 2x2 pixels, Y:U:V sampled as 4:1:1
        // Calculate address indices for the required 4 Y values
        DataType srcYIdx[YUV_PER_DEAL_NUM];
        ComputeSrcIdx(srcYIdx, coord, cropStartPosH, cropStartPosW, inputSizeW, inputSizeH);
        // Calculate address index for the required 1 U value (V is adjacent to U)
        DataType srcUIdx = coord.nIdx * inputSizeH * inputSizeW * 3 / 2 + inputSizeH * inputSizeW +
                           ((cropStartPosH + DIGIT_TWO * coord.hIdx) >> 1) * inputSizeW +
                           (cropStartPosW + DIGIT_TWO * coord.wIdx);
        DataType srcVIdx = srcUIdx + 1;

        for (uint8_t i = 0; i < YUV_PER_DEAL_NUM; i++) {
            DataType srcYIdxIndex = srcYIdx[i];
            RgbPack<DataType> dstRgbIdxIndex = dstRgbIdx[i];

            RgbPack<uint8_t> dstRgb;
            Yuv2Rgb(dstRgb, yuvGM[srcYIdxIndex], yuvGM[srcUIdx], yuvGM[srcVIdx], tD.cscParam);

            DataConversion(outputGM[dstRgbIdxIndex.r], dstRgb.r, tD.dtcParam, CHANNEL_NUM_0);
            DataConversion(outputGM[dstRgbIdxIndex.g], dstRgb.g, tD.dtcParam, CHANNEL_NUM_1);
            DataConversion(outputGM[dstRgbIdxIndex.b], dstRgb.b, tD.dtcParam, CHANNEL_NUM_2);
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t batchSize = this->tilingData_.batchNum *
                        (this->tilingData_.cropParam.cropSizeH >> 1) *
                        (this->tilingData_.cropParam.cropSizeW >> 1);

    Simt::VF_CALL<Aipp_Kernel::SimtComputeYuv<T, DataType>>(Simt::Dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, batchSize);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_H