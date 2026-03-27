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

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_RGB_H
#define AIPP_OP_KERNEL_ARCH35_YUV_RGB_H

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
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void ComputeDstIdx(
    RgbPack<DataType>* dstRgbIdx, const CoordPack<DataType>& coord, const AippTilingData& tD)
{
    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) + 
                                DIGIT_2 * coord.hIdx * tD.outputSizeW  + DIGIT_2 * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + tD.outputSizeH * tD.outputSizeW;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + tD.outputSizeH * tD.outputSizeW;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + 1;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                    (DIGIT_2 * coord.hIdx + 1) * tD.outputSizeW + DIGIT_2 * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + tD.outputSizeH * tD.outputSizeW;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + tD.outputSizeH * tD.outputSizeW;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + 1;
    } else {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) + 
                                DIGIT_2 * coord.hIdx * tD.outputSizeW * CHANNEL_THREE +
                                DIGIT_2 * coord.wIdx * CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + CHANNEL_THREE;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                    (DIGIT_2 * coord.hIdx + 1) * tD.outputSizeW * CHANNEL_THREE +
                                    DIGIT_2 * coord.wIdx * CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + CHANNEL_THREE;
    }
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

__aicore__ __attribute__((always_inline)) inline bool IsPixelInPaddingForYuv(
    uint32_t pixelH, uint32_t pixelW, const AippTilingData& tD, bool allEvenPadding, bool blockAllInPadding)
{
    if (tD.paddingParam.paddingSwitch == 0) {
        return false;
    }
    if (allEvenPadding) {
        return blockAllInPadding;
    }
    return (pixelH < tD.paddingParam.topPaddingSize) ||
           (pixelH >= tD.paddingParam.topPaddingSize + tD.cropParam.cropSizeH) ||
           (pixelW < tD.paddingParam.leftPaddingSize) ||
           (pixelW >= tD.paddingParam.leftPaddingSize + tD.cropParam.cropSizeW);
}

template <typename T, typename DataType>
__aicore__ __attribute__((always_inline)) inline void ProcessYuvPixel(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const RgbPack<DataType> dstIdx,
    uint32_t nIdx, uint32_t croodH, uint32_t croodW, const AippTilingData& tD)
{
    uint32_t srcYIdx = nIdx * tD.inputSizeH * tD.inputSizeW * 3 / 2 + 
                        (tD.cropParam.cropStartPosH + croodH) * tD.inputSizeW +
                        (tD.cropParam.cropStartPosW + croodW);
    uint32_t srcUIdx = nIdx * tD.inputSizeH * tD.inputSizeW * 3 / 2 + tD.inputSizeH * tD.inputSizeW +
                        ((tD.cropParam.cropStartPosH + (croodH & ~1)) >> 1) * tD.inputSizeW +
                        (tD.cropParam.cropStartPosW + (croodW & ~1));
    uint32_t srcVIdx = srcUIdx + 1;
    RgbPack<uint8_t> dstRgb;
    Yuv2Rgb(dstRgb, yuvGM[srcYIdx], yuvGM[srcUIdx], yuvGM[srcVIdx], tD.cscParam);

    DataConversion(outputGM[dstIdx.r], dstRgb.r, tD.dtcParam, CHANNEL_NUM_0);
    DataConversion(outputGM[dstIdx.g], dstRgb.g, tD.dtcParam, CHANNEL_NUM_1);
    DataConversion(outputGM[dstIdx.b], dstRgb.b, tD.dtcParam, CHANNEL_NUM_2);
}

template <typename T, typename DataType>
__aicore__ __attribute__((always_inline)) inline void ProcessYuvBlock(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM,
    const RgbPack<DataType>* dstRgbIdx, const CoordPack<DataType>& coord,
    const AippTilingData& tD, float padValue, bool allEvenPadding, bool blockAllInPadding)
{
    uint32_t actualH = coord.hIdx * DIGIT_2;
    uint32_t actualW = coord.wIdx * DIGIT_2;

    for (uint8_t i = 0; i < YUV_PER_DEAL_NUM; i++) {
        RgbPack<DataType> dstRgbIdxIndex = dstRgbIdx[i];
        uint32_t pixelH = (i >= YUV_DEAL_NUM_2) ? actualH + 1 : actualH;
        uint32_t pixelW = (i == YUV_DEAL_NUM_1 || i == YUV_DEAL_NUM_3) ? actualW + 1 : actualW;
        if (IsPixelInPaddingForYuv(pixelH, pixelW, tD, allEvenPadding, blockAllInPadding)) {
            AssignPadValue(outputGM[dstRgbIdxIndex.r], padValue);
            AssignPadValue(outputGM[dstRgbIdxIndex.g], padValue);
            AssignPadValue(outputGM[dstRgbIdxIndex.b], padValue);
        } else {
            uint32_t cropCoordH = pixelH - tD.paddingParam.topPaddingSize;
            uint32_t cropCoordW = pixelW - tD.paddingParam.leftPaddingSize;
            ProcessYuvPixel<T, DataType>(yuvGM, outputGM, dstRgbIdxIndex, coord.nIdx,
                                         cropCoordH, cropCoordW, tD);
        }
    }
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void SimtComputeYuv(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;
    int32_t paddingSwitch = tD.paddingParam.paddingSwitch;
    int32_t leftPaddingSize = tD.paddingParam.leftPaddingSize;
    int32_t rightPaddingSize = tD.paddingParam.rightPaddingSize;
    int32_t topPaddingSize = tD.paddingParam.topPaddingSize;
    int32_t bottomPaddingSize = tD.paddingParam.bottomPaddingSize;
    float padValue = tD.paddingParam.padValue;

    bool allEvenPadding = (topPaddingSize % DIGIT_2 == 0) &&
                          (bottomPaddingSize % DIGIT_2 == 0) &&
                          (leftPaddingSize % DIGIT_2 == 0) &&
                          (rightPaddingSize % DIGIT_2 == 0);

    for (DataType idx = Simt::GetThreadIdx() + blockIdx * Simt::GetThreadNum(); idx < batchSize;
         idx += blockNum * Simt::GetThreadNum()) {
        CoordPack<DataType> coord;
        coord.nIdx = idx / ((outputSizeH >> 1) * (outputSizeW >> 1));
        DataType newIdx = idx - coord.nIdx * ((outputSizeH >> 1) * (outputSizeW >> 1));
        coord.hIdx = newIdx / (outputSizeW >> 1);
        coord.wIdx = newIdx - coord.hIdx * (outputSizeW >> 1);

        RgbPack<DataType> dstRgbIdx[YUV_PER_DEAL_NUM];
        ComputeDstIdx(dstRgbIdx, coord, tD);

        uint32_t actualH = coord.hIdx * DIGIT_2;
        uint32_t actualW = coord.wIdx * DIGIT_2;

        bool blockAllInPadding = (paddingSwitch != 0 && allEvenPadding) &&
                                 ((actualH < topPaddingSize) ||
                                 (actualH >= topPaddingSize + tD.cropParam.cropSizeH) ||
                                 (actualW < leftPaddingSize) ||
                                 (actualW >= leftPaddingSize + tD.cropParam.cropSizeW));

        ProcessYuvBlock<T, DataType>(yuvGM, outputGM, dstRgbIdx, coord, tD,
                                     padValue, allEvenPadding, blockAllInPadding);
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t batchSize = this->tilingData_.batchNum *
                         ((this->tilingData_.outputSizeH) >> 1) * ((this->tilingData_.outputSizeW) >> 1);

    Simt::VF_CALL<Aipp_Kernel::SimtComputeYuv<T, DataType>>(Simt::Dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, batchSize);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_RGB_H