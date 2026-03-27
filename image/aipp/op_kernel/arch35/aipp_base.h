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
 * \file aipp_base.h
 * \brief aipp kernel base
 */
#ifndef AIPP_OP_KERNEL_ARCH35_BASE_H
#define AIPP_OP_KERNEL_ARCH35_BASE_H

#include "kernel_operator.h"
#include "aipp_struct.h"

#define CLIP3(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

namespace Aipp_Kernel {
using namespace AscendC;
constexpr uint32_t MAX_THREAD_NUM = 256;
constexpr uint32_t CSC_MATRIX_SCALE = 512;
constexpr uint32_t MAX_UINT8 = 255;
constexpr uint8_t CHANNEL_THREE = 3;
constexpr uint8_t YUV_PER_DEAL_NUM = 4;
constexpr uint8_t YUV_DEAL_NUM_0 = 0;
constexpr uint8_t YUV_DEAL_NUM_1 = 1;
constexpr uint8_t YUV_DEAL_NUM_2 = 2;
constexpr uint8_t YUV_DEAL_NUM_3 = 3;
constexpr uint8_t CHANNEL_NUM_0 = 0;
constexpr uint8_t CHANNEL_NUM_1 = 1;
constexpr uint8_t CHANNEL_NUM_2 = 2;
constexpr uint8_t DIGIT_2 = 2;
constexpr uint8_t DIGIT_3 = 3;
constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;
constexpr uint8_t XRGB8888_U8_FORMAT = 3;

template <typename T, typename DataType>
class AippBase {
public:
    __aicore__ inline AippBase(){};
    __aicore__ inline void BaseInit(const AippTilingData& tilingData);

public:
    AippTilingData tilingData_ = {};

    uint64_t totalNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
};

template <typename T, typename DataType>
__aicore__ inline void AippBase<T, DataType>::BaseInit(const AippTilingData& tilingData)
{
    tilingData_ = tilingData;

    blockNum_ = GetBlockNum();
    blockIdx_ = GetBlockIdx();
    totalNum_ = tilingData_.batchNum * tilingData_.outputSizeH * tilingData_.outputSizeW;
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void DataConversion(
    T& dst, uint8_t src, const DtcParam& dtcParam, int32_t channelIndex)
{
    if constexpr (sizeof(T) == DIGIT_2) {
        if (channelIndex == 0) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn0) - static_cast<T>(dtcParam.dtcPixelMinChn0)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn0);
        } else if (channelIndex == 1) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn1) - static_cast<T>(dtcParam.dtcPixelMinChn1)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn1);
        } else if (channelIndex == 2) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn2) - static_cast<T>(dtcParam.dtcPixelMinChn2)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn2);
        } else if (channelIndex == 3) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn3) - static_cast<T>(dtcParam.dtcPixelMinChn3)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn3);
        }
    } else {
        dst = src;
    }
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void AssignPadValue(T& dst, float padValue)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        dst = static_cast<T>(CLIP3(padValue, 0.0f, 255.0f));
    } else {
        dst = static_cast<T>(padValue);
    }
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void RgbComputeDstIdx(
    RgbPack<DataType> &dstIdx, const CoordPack<DataType>& coord, const AippTilingData& tD)
{
    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        dstIdx.r = coord.nIdx * tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE +
                   coord.hIdx * tD.outputSizeW  + coord.wIdx;
        dstIdx.g = dstIdx.r + tD.outputSizeH * tD.outputSizeW;
        dstIdx.b = dstIdx.g + tD.outputSizeH * tD.outputSizeW;
    } else {
        dstIdx.r = coord.nIdx * tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE +
                   coord.hIdx * tD.outputSizeW * CHANNEL_THREE + coord.wIdx * CHANNEL_THREE;
        dstIdx.g = dstIdx.r + 1;
        dstIdx.b = dstIdx.g + 1;
    }
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void RgbComputeSrcIdx(
    RgbPack<DataType> &srcIdx, const CoordPack<DataType>& coord, const AippTilingData& tD,
    const DataType offset = 0)
{
    srcIdx.r = coord.nIdx * tD.inputSizeH * tD.inputSizeW * tD.channelNum +
        (tD.cropParam.cropStartPosH + coord.hIdx - tD.paddingParam.topPaddingSize) * tD.inputSizeW * tD.channelNum +
        (tD.cropParam.cropStartPosW + coord.wIdx - tD.paddingParam.leftPaddingSize) * tD.channelNum + offset;
    srcIdx.g = srcIdx.r + 1;
    srcIdx.b = srcIdx.g + 1;
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline bool IsPixelInPadding(
    DataType hIdx, DataType wIdx, const AippTilingData& tD)
{
    if (tD.paddingParam.paddingSwitch == 0) {
        return false;
    }
    int32_t leftPaddingSize = tD.paddingParam.leftPaddingSize;
    int32_t topPaddingSize = tD.paddingParam.topPaddingSize;
    uint32_t cropSizeH = tD.cropParam.cropSizeH;
    uint32_t cropSizeW = tD.cropParam.cropSizeW;
    return (hIdx < topPaddingSize) ||
           (hIdx >= topPaddingSize + cropSizeH) ||
           (wIdx < leftPaddingSize) ||
           (wIdx >= leftPaddingSize + cropSizeW);
}

template <typename DataType>
__aicore__ __attribute__((always_inline)) inline void ComputeCoordFromIndex(
    DataType idx, uint32_t outputSizeH, uint32_t outputSizeW,
    CoordPack<DataType>& coord)
{
    coord.nIdx = idx / (outputSizeH * outputSizeW);
    DataType newIdx = idx - coord.nIdx * outputSizeH * outputSizeW;
    coord.hIdx = newIdx / outputSizeW;
    coord.wIdx = newIdx - coord.hIdx * outputSizeW;
}
} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_BASE_H