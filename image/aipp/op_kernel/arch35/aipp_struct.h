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

struct AippTilingData {
    uint8_t inputFormat = 0;
    uint8_t imageFormat = 0;
    uint32_t batchNum = 1;
    uint32_t channelNum = 3;
    uint32_t inputSizeW = 0;
    uint32_t inputSizeH = 0;
    uint8_t cropSwitch = 0;
    uint32_t cropStartPosH = 0;
    uint32_t cropStartPosW = 0;
    uint32_t cropSizeH = 0;
    uint32_t cropSizeW = 0;

    int16_t dtcPixelMeanChn0 = 0;
    int16_t dtcPixelMeanChn1 = 0;
    int16_t dtcPixelMeanChn2 = 0;
    int16_t dtcPixelMeanChn3 = 0;
    float dtcPixelMinChn0 = 0.0;
    float dtcPixelMinChn1 = 0.0;
    float dtcPixelMinChn2 = 0.0;
    float dtcPixelMinChn3 = 0.0;
    float dtcPixelVarReciChn0 = 1.0;
    float dtcPixelVarReciChn1 = 1.0;
    float dtcPixelVarReciChn2 = 1.0;
    float dtcPixelVarReciChn3 = 1.0;

    uint8_t cscSwitch = 0;
    int16_t cscMatrix00 = 0;
    int16_t cscMatrix01 = 0;
    int16_t cscMatrix02 = 0;
    int16_t cscMatrix10 = 0;
    int16_t cscMatrix11 = 0;
    int16_t cscMatrix12 = 0;
    int16_t cscMatrix20 = 0;
    int16_t cscMatrix21 = 0;
    int16_t cscMatrix22 = 0;

    uint8_t outBias0 = 0;
    uint8_t outBias1 = 0;
    uint8_t outBias2 = 0;

    uint8_t inBias0 = 0;
    uint8_t inBias1 = 0;
    uint8_t inBias2 = 0;
};

#endif // AIPP_OP_KERNEL_ARCH35_STRUCT_H