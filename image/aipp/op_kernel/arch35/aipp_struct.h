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
 * \file aipp_struct.h
 * \brief aipp struct
 */

#ifndef AIPP_OP_KERNEL_ARCH35_STRUCT_H
#define AIPP_OP_KERNEL_ARCH35_STRUCT_H

namespace Aipp_Kernel {
struct CscParam {
    int16_t cscSwitch = 0;

    int16_t rbuvSwapSwitch = 0;
    int16_t axSwapSwitch = 0;

    int16_t cscMatrix00 = 298;
    int16_t cscMatrix01 = 516;
    int16_t cscMatrix02 = 0;
    int16_t cscMatrix10 = 298;
    int16_t cscMatrix11 = -100;
    int16_t cscMatrix12 = -208;
    int16_t cscMatrix20 = 298;
    int16_t cscMatrix21 = 0;
    int16_t cscMatrix22 = 409;
    int16_t outBias0 = 16;
    int16_t outBias1 = 128;
    int16_t outBias2 = 128;
    int16_t inBias0 = 16;
    int16_t inBias1 = 128;
    int16_t inBias2 = 128;
};

struct CropParam {
    int16_t cropSwitch = 0;

    uint32_t cropStartPosH = 0;
    uint32_t cropStartPosW = 0;
    uint32_t cropSizeH = 0;
    uint32_t cropSizeW = 0;
};

struct DtcParam {
    int16_t dtcPixelMeanChn0 = 0;
    int16_t dtcPixelMeanChn1 = 0;
    int16_t dtcPixelMeanChn2 = 0;
    int16_t dtcPixelMeanChn3 = 0;
    float dtcPixelMinChn0 = 0;
    float dtcPixelMinChn1 = 0;
    float dtcPixelMinChn2 = 0;
    float dtcPixelMinChn3 = 0;
    float dtcPixelVarReciChn0 = 1;
    float dtcPixelVarReciChn1 = 1;
    float dtcPixelVarReciChn2 = 1;
    float dtcPixelVarReciChn3 = 1;
};

struct PaddingParam {
    int32_t paddingSwitch = 0;
    int32_t leftPaddingSize = 0;
    int32_t rightPaddingSize = 0;
    int32_t topPaddingSize = 0;
    int32_t bottomPaddingSize = 0;
    float padValue = 0.0f;
};

struct AippTilingData {
    uint8_t imageFormat = 0;
    uint8_t outputFormat = 0;

    uint32_t batchNum = 0;
    uint32_t channelNum = 0;
    uint32_t inputSizeW = 0;
    uint32_t inputSizeH = 0;
    uint32_t outputSizeW = 0;
    uint32_t outputSizeH = 0;

    CscParam cscParam;
    CropParam cropParam;
    DtcParam dtcParam;
    PaddingParam paddingParam;
};

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
} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_STRUCT_H