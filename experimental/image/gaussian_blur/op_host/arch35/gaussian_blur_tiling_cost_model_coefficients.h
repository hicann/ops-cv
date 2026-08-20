/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Generated GaussianBlur tiling coefficients. Do not edit manually. */
#ifndef GAUSSIAN_BLUR_TILING_COST_MODEL_COEFFICIENTS_H
#define GAUSSIAN_BLUR_TILING_COST_MODEL_COEFFICIENTS_H

#include <array>
#include <cstdint>

namespace optiling::gaussian_blur_cost_model {

constexpr uint32_t LEARNED_FEATURE_COUNT = 13U;
using LearnedCoefficients = std::array<double, LEARNED_FEATURE_COUNT>;

constexpr LearnedCoefficients LEARNED_DIRECT_COEFFICIENTS = {
    4.3610205819780381, 0, 0, 0, 0, 10.111809402507291, 0, 0, 2.5617619295553573, 0, 0, 0, 0,
};

constexpr LearnedCoefficients LEARNED_C1_K31_COEFFICIENTS = {
    3.4765900956552773, 0, 0.64154579996816985, 0, 0, 2.8591585851197197, 0, 0, 16.813873097816046, 0, 0, 0, 0,
};

constexpr LearnedCoefficients LEARNED_C1_TILE_COEFFICIENTS = {
    4.254659339595614,  3.686405201347954, 0, 0, 0, 1.7038772580545374, 0, 0,
    10.502708070091613, 10.3844166929597,  0, 0, 0,
};

constexpr LearnedCoefficients LEARNED_C8_RING_COEFFICIENTS = {
    5.7416148047299433,
    12.391224735284196,
    0,
    336.00184129247538,
    0,
    0,
    238.83756349936678,
    0,
    0,
    0,
    0,
    1.0375168306220812,
    0,
};

constexpr LearnedCoefficients LEARNED_MULTI_C8_COEFFICIENTS = {
    4.4622153567889242,
    7.9288823794141301,
    30.516669794461901,
    125.36106923202111,
    0,
    0,
    152.15565026830416,
    96.482738789882944,
    2.5830898196641021,
    0,
    1.1807113050802152,
    0,
    0,
};

} // namespace optiling::gaussian_blur_cost_model

#endif // GAUSSIAN_BLUR_TILING_COST_MODEL_COEFFICIENTS_H
