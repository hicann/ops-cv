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
constexpr uint32_t MAX_TGHREAD_NUM = 256;
constexpr uint32_t CSC_MATRIX_SCALE = 512;
constexpr uint8_t MAX_CHANNEL_NUM = 4;
constexpr uint8_t RGB_CHANNEL_NUM = 3;
constexpr uint8_t YUV_CHANNEL_NUM = 3;
constexpr uint8_t YUV_PER_DEAL_NUM = 4;
constexpr uint8_t YUV_DEAL_NUM_0 = 0;
constexpr uint8_t YUV_DEAL_NUM_1 = 1;
constexpr uint8_t YUV_DEAL_NUM_2 = 2;
constexpr uint8_t YUV_DEAL_NUM_3 = 3;
constexpr uint8_t CHANNEL_NUM_0 = 0;
constexpr uint8_t CHANNEL_NUM_1 = 1;
constexpr uint8_t CHANNEL_NUM_2 = 2;
constexpr uint8_t DIGIT_TWO = 2;
constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;

template <typename T>
struct RgbPack {
    T r = 0;
    T g = 0;
    T b = 0;
};

template <typename T>
struct YuvPack {
    T y = 0;
    T u = 0;
    T v = 0;
};

template <typename T>
struct CoordPack {
    T nIdx = 0;
    T hIdx = 0;
    T wIdx = 0;
};

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
    totalNum_ = tilingData_.batchNum * tilingData_.cropParam.cropSizeH * tilingData_.cropParam.cropSizeW;
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void DataConversion(
    T& dst, uint8_t src, const DtcParam& dtcParam, int32_t channelIndex)
{
    if constexpr (sizeof(T) == DIGIT_TWO) {
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
            // 4th channel is reserved
            dst = 0;
        }
    } else {
        dst = src;
    }
}

} // namespace Aipp_Kernel

#endif // AIPP_OP_KERNEL_ARCH35_BASE_H