/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_UTILS_H_
#define GAUSSIAN_BLUR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gaussian_blur {

static constexpr uint32_t MAX_KERNEL_SIZE = 255U;
static constexpr int64_t BORDER_CONSTANT = 0;
static constexpr int64_t BORDER_REPLICATE = 1;
static constexpr int64_t BORDER_REFLECT = 2;
static constexpr int64_t BORDER_REFLECT_101 = 4;
static constexpr int64_t BORDER_DEFAULT = BORDER_REFLECT_101;
static constexpr int64_t BORDER_ISOLATED = 16;
static constexpr double SIGMA_INFER_SCALE_FLOAT = 4.0;

struct CanonicalParams {
    int64_t kernelW = 1;
    int64_t kernelH = 1;
    double sigmaX = 0.0;
    double sigmaY = 0.0;
    int64_t borderType = BORDER_REPLICATE;
};

inline bool IsExplicitKernelSizeValid(int64_t value) { return value >= 0 && (value == 0 || ((value & 1LL) == 1LL)); }

inline bool IsRuntimeKernelSizeSupported(int64_t value)
{
    return value > 0 && ((value & 1LL) == 1LL) && value <= static_cast<int64_t>(MAX_KERNEL_SIZE);
}

inline bool InferKernelSizeFromSigma(double sigma, int64_t& inferredKernel)
{
    if (!(sigma > 0.0)) {
        return false;
    }
    inferredKernel = static_cast<int64_t>(std::llround(sigma * SIGMA_INFER_SCALE_FLOAT * 2.0 + 1.0));
    inferredKernel |= 1LL;
    if (inferredKernel < 1) {
        inferredKernel = 1;
    }
    return true;
}

inline bool CanonicalizeBorderType(int64_t borderType, int64_t& canonicalBorderType)
{
    if ((borderType & BORDER_ISOLATED) != 0) {
        return false;
    }
    canonicalBorderType = borderType & ~BORDER_ISOLATED;
    if (canonicalBorderType == BORDER_DEFAULT) {
        canonicalBorderType = BORDER_REFLECT_101;
    }
    return canonicalBorderType == BORDER_CONSTANT || canonicalBorderType == BORDER_REPLICATE ||
           canonicalBorderType == BORDER_REFLECT || canonicalBorderType == BORDER_REFLECT_101;
}

inline bool CanonicalizeParams(int64_t kernelW, int64_t kernelH, double sigmaX, double sigmaY, int64_t borderType,
                               uint32_t width, uint32_t height, CanonicalParams& canonical)
{
    int64_t canonicalBorderType = BORDER_REPLICATE;
    if (!CanonicalizeBorderType(borderType, canonicalBorderType)) {
        return false;
    }

    const double finalSigmaX = sigmaX;
    const double finalSigmaY = sigmaY <= 0.0 ? sigmaX : sigmaY;
    int64_t finalKernelW = kernelW;
    int64_t finalKernelH = kernelH;

    if (finalKernelW == 0 && !InferKernelSizeFromSigma(finalSigmaX, finalKernelW)) {
        return false;
    }
    if (finalKernelH == 0 && !InferKernelSizeFromSigma(finalSigmaY, finalKernelH)) {
        return false;
    }

    if (canonicalBorderType != BORDER_CONSTANT) {
        if (width == 1U) {
            finalKernelW = 1;
        }
        if (height == 1U) {
            finalKernelH = 1;
        }
    }

    if (!IsRuntimeKernelSizeSupported(finalKernelW) || !IsRuntimeKernelSizeSupported(finalKernelH)) {
        return false;
    }

    canonical.kernelW = finalKernelW;
    canonical.kernelH = finalKernelH;
    canonical.sigmaX = std::max(finalSigmaX, 0.0);
    canonical.sigmaY = std::max(finalSigmaY, 0.0);
    canonical.borderType = canonicalBorderType;
    return true;
}

inline bool TryBuildSmallSigmaZeroKernel(uint32_t kernel, float* weight)
{
    if (kernel == 1U) {
        weight[0] = 1.0f;
        return true;
    }
    if (kernel == 3U) {
        const float table[] = {0.25f, 0.5f, 0.25f};
        std::copy(table, table + 3, weight);
        return true;
    }
    if (kernel == 5U) {
        const float table[] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
        std::copy(table, table + 5, weight);
        return true;
    }
    if (kernel == 7U) {
        const float table[] = {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f};
        std::copy(table, table + 7, weight);
        return true;
    }
    if (kernel == 9U) {
        const float table[] = {4.0f / 256.0f,  13.0f / 256.0f, 30.0f / 256.0f, 51.0f / 256.0f, 60.0f / 256.0f,
                               51.0f / 256.0f, 30.0f / 256.0f, 13.0f / 256.0f, 4.0f / 256.0f};
        std::copy(table, table + 9, weight);
        return true;
    }
    return false;
}

inline void BuildGaussianWeights(uint32_t kernel, double sigma, float* weight, uint32_t maxWeightSize)
{
    for (uint32_t i = 0; i < maxWeightSize; ++i) {
        weight[i] = 0.0f;
    }
    if (kernel == 0U || kernel > maxWeightSize) {
        return;
    }
    if (sigma <= 0.0 && TryBuildSmallSigmaZeroKernel(kernel, weight)) {
        return;
    }

    // Match OpenCV's FP32 kernel data flow: compute one side, mirror it,
    // normalize in double precision, and cast the coefficients to float.
    const double finalSigma = sigma > 0.0 ? sigma : (static_cast<double>(kernel) * 0.15 + 0.35);
    const double scale2X = -0.125 / (finalSigma * finalSigma);
    const uint32_t half = (kernel - 1U) / 2U;
    double sideValues[MAX_KERNEL_SIZE] = {};
    double sum = 0.0;
    int32_t x = 1 - static_cast<int32_t>(kernel);
    for (uint32_t i = 0; i < half; ++i, x += 2) {
        const double value = std::exp(static_cast<double>(x * x) * scale2X);
        sideValues[i] = value;
        sum += value;
    }

    sum = sum * 2.0 + 1.0;
    if ((kernel & 1U) == 0U) {
        sum += 1.0;
    }
    if (sum <= 0.0) {
        return;
    }

    const double multiplier = 1.0 / sum;
    for (uint32_t i = 0; i < half; ++i) {
        const float value = static_cast<float>(sideValues[i] * multiplier);
        weight[i] = value;
        weight[kernel - 1U - i] = value;
    }
    weight[half] = static_cast<float>(multiplier);
    if ((kernel & 1U) == 0U && half + 1U < kernel) {
        weight[half + 1U] = weight[half];
    }
}

} // namespace gaussian_blur

#endif // GAUSSIAN_BLUR_UTILS_H_
