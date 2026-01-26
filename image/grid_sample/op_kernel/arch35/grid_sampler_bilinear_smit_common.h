/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grid_sampler_bilinear_smit_common.h
 * \brief grid sampler kernel info
 */
#ifndef GRID_SAMPLER_BILINEAR_SIMT_COMMON_H
#define GRID_SAMPLER_BILINEAR_SIMT_COMMON_H
#include "kernel_operator.h"
namespace GridSample {

using namespace AscendC;

const uint32_t REFLECT_RATIO_95 = 2;
const uint32_t PADDING_MODE_ZEROS_95 = 0;
const uint32_t PADDING_MODE_BORDER_95 = 1;
const uint32_t PADDING_MODE_REFLECTION_95 = 2;
const uint32_t INT_MAX = 2147483647;
const uint32_t INT_MIN = -2147483648;
const float DEFAULT_FAULT_VALUE = -100.0f;
const uint32_t ALIGNCORNER_FALSE = 0;
const uint32_t ALIGNCORNER_TRUE = 1;
const uint32_t VF_MAX_THREAD_NUM = 1024;
const uint32_t VF_MAX_THREAD_NUM_3D = 512;


__aicore__ __attribute__((always_inline)) inline float ClipCoordinates(float coord, int32_t size)
{
    coord = coord < 0 ? 0 : coord;
    coord = coord > (size - 1) ? (float)size - 1 : coord;
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float reflectCoordinates(float coord, int twiceLow, int32_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        return 0;
    }
    float min = static_cast<float>(twiceLow) / 2;
    float span = static_cast<float>(twiceHigh - twiceLow) / 2;
    coord = Simt::Abs(coord - min);
    float extra = Simt::Mod(coord, span);
    int32_t flips = static_cast<int32_t>(Simt::Floor(coord / span));
    if (flips % REFLECT_RATIO_95 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float safeDowngradeToIntRange(float coord)
{
    if (!Simt::IsFinite(coord)) {
        return DEFAULT_FAULT_VALUE;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float Clip(
    float coord, int32_t size, int32_t paddingMode, int32_t alignCorners)
{
    if (paddingMode == PADDING_MODE_BORDER_95) {
        coord = ClipCoordinates(coord, size);
    } else if (paddingMode == PADDING_MODE_REFLECTION_95) {
        if (alignCorners) {
            coord = reflectCoordinates(coord, 0, REFLECT_RATIO_95 * (size - 1));
        } else {
            coord = reflectCoordinates(coord, -1, REFLECT_RATIO_95 * size - 1);
        }
        coord = ClipCoordinates(coord, size);
    }
    safeDowngradeToIntRange(coord);
    return coord;
}

} // namespace GridSample
#endif // GRID_SAMPLER_BILINEAR_SIMT_COMMON_H