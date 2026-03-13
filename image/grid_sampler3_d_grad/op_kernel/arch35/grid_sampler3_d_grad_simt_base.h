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
 * \file grid_sampler3_d_grad_simt.h
 * \brief
 */
#ifndef GRID_SAMPLER3D_GRAD_SIMT_BASE_H_
#define GRID_SAMPLER3D_GRAD_SIMT_BASE_H_

#include "simt_api/asc_simt.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace GridSampler3DGradSimtBase {
    constexpr int32_t INT_MAX = 2147483647;
    constexpr int32_t INT_MIN = -2147483648;
    constexpr int32_t INPUT_NUM = 3;
    constexpr int32_t OUTPUT_NUM = 2;
    constexpr int32_t GRAD_INPUT_INDEX = 0;
    constexpr int32_t X_INPUT_INDEX = 1;
    constexpr int32_t GRID_INPUT_INDEX = 2;
    constexpr int32_t DX_INPUT_INDEX = 3;
    constexpr int32_t DGRID_INPUT_INDEX = 4;
    constexpr int32_t WORKSPACE_INDEX = 5;
    constexpr int32_t TMP_OUT_INDEX = 0;
    constexpr int32_t GM_PARAMS_SIZE = 6;
    constexpr int32_t DX_OUTPUT_INDEX = 0;
    constexpr int32_t DGRID_OUTPUT_INDEX = 1;
    constexpr int32_t BILINEAR = 0;
    constexpr int32_t NEAREST = 1;
    constexpr int32_t BORDER = 1;
    constexpr int32_t REFLECTION = 2;
    constexpr float DEFAULT_FAULT_VALUE = -100.0f;
    constexpr uint32_t VF_MAX_THREAD_NUM = 256;

__aicore__ __attribute__((always_inline)) inline float UnnormallizeSetGrad(
    float coord, uint32_t size, uint32_t padding, uint32_t alignCorners, float* gradInValue)
{
    if (alignCorners == 1) {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        coord = (coord + 1) / 2 * (size - 1);
        *gradInValue = static_cast<float>(size - 1) / 2;
    } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        coord = ((coord + 1) * size - 1) / 2;
        *gradInValue = static_cast<float>(size) / 2;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float ClipCoorDinatesSetGrad(
    float coord, uint32_t clip_limit, float* gradClipValue)
{
    if (coord <= static_cast<float>(0)) {
        *gradClipValue = static_cast<float>(0);
        return static_cast<float>(0);
    } else {
        float max = static_cast<float>(clip_limit - 1);
        if (coord >= max) {
            *gradClipValue = static_cast<float>(0);
            return max;
        } else {
            *gradClipValue = static_cast<float>(1);
            return coord;
        }
    }
}

__aicore__ __attribute__((always_inline)) inline float ReflectCoordinatesSetGrad(
    float coord, int twiceLow, uint32_t twiceHigh, float* gradReflValue)
{
    if (twiceLow == twiceHigh) {
        *gradReflValue = static_cast<float>(0);
        return static_cast<float>(0);
    }
    float gradInMult = 0;
    float min = static_cast<float>(twiceLow) / 2;
    float span = static_cast<float>(twiceHigh - twiceLow) / 2;
    coord = coord - min;
    if (coord < static_cast<float>(0)) {
        gradInMult = static_cast<float>(-1);
        coord = -coord;
    } else {
        gradInMult = static_cast<float>(1);
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    float extra = Simt::Mod(coord, span);
    int32_t flips = static_cast<int32_t>(Simt::Floor(coord / span));
    if (flips % 2 == 0) {
        *gradReflValue = gradInMult;
        return extra + min;
    } else {
        *gradReflValue = -gradInMult;
        return span - extra + min;
    }
}

__aicore__ __attribute__((always_inline)) inline float SafeDowngradeToIntRange(float coord)
{
    if (!Simt::IsFinite(coord)) {
        return DEFAULT_FAULT_VALUE;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline int32_t GetFloorValue(float x)
{
    float negativeValue = static_cast<float>(0.0);
    float floorFactor = static_cast<float>(-1);
    return (x >= negativeValue ? static_cast<int32_t>(x) : static_cast<int32_t>(floorFactor + x));
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void GetGradOutPointValueAndDxIndex(
    __gm__ T* gradOutGmAddr, int32_t inputXDepth, int32_t inputXHeight, int32_t inputXWidth, uint32_t gridD,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t depthCol, uint32_t heightCol, uint32_t widthCol,
    uint32_t channelIndex, uint32_t gridoffsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW, uint32_t channel,
    uint32_t newInputIndex, float* gradOutValue, uint32_t* dxIndex)
{
    if (inputXDepth >= 0 && inputXHeight >= 0 && inputXWidth >= 0 && inputXDepth < xD && inputXHeight < xH &&
        inputXWidth < xW) {
        uint32_t gradOutValueIndex = batchNum * channel * gridD * gridH * gridW + channelIndex * gridD * gridH * gridW +
                                     depthCol * gridH * gridW + heightCol * gridW + widthCol;
        *gradOutValue = static_cast<float>(gradOutGmAddr[gradOutValueIndex]);
        *dxIndex = static_cast<uint32_t>(
            newInputIndex + channelIndex * xD * xH * xW + inputXDepth * xH * xW + inputXHeight * xW + inputXWidth);
    }
}

__aicore__ __attribute__((always_inline)) inline float ComputeSourceIndexSetGrad(
    float coord, uint32_t size, uint32_t padding, uint32_t alignCorners, float* gradInValue)
{
    float gradClipValue = 0;
    float gradReflValue = 0;
    coord = UnnormallizeSetGrad(coord, size, padding, alignCorners, gradInValue);

    if (padding == BORDER) {
        // clip coordinates to image borders
        coord = ClipCoorDinatesSetGrad(coord, size, &gradClipValue);
        *gradInValue = (*gradInValue) * gradClipValue;
    } else if (padding == REFLECTION) {
        // reflect coordinates by image borders
        if (alignCorners) {
            coord = ReflectCoordinatesSetGrad(coord, 0, 2 * (size - 1), &gradReflValue);
        } else {
            coord = ReflectCoordinatesSetGrad(coord, -1, 2 * size - 1, &gradReflValue);
        }
        // clip coordinates to image borders
        coord = ClipCoorDinatesSetGrad(coord, size, &gradClipValue);
        *gradInValue = (*gradInValue) * gradReflValue * gradClipValue;
    }
    coord = SafeDowngradeToIntRange(coord);
    return coord;
}
} // namespace GridSampler3DGradSimtBase
#endif // GRID_SAMPLER3D_GRAD_SIMT_BASE_H_
