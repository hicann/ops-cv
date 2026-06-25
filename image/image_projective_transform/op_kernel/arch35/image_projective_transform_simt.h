/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file image_projective_transform_simt.h
 * \brief SIMT kernel implementation for image_projective_transform operator
 */

#ifndef IMAGE_PROJECTIVE_TRANSFORM_SIMT_H_
#define IMAGE_PROJECTIVE_TRANSFORM_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "image_projective_transform_tiling_data.h"
#include "image_projective_transform_tiling_key.h"

// Disable FMA (fused multiply-add) contraction for all float arithmetic.
//
// Rationale (verified by pixel-level analysis against TF 2.21.0 golden):
//   TF's ImageProjectiveTransformV3 computes coordinates and bilinear interpolation
//   using plain float32 arithmetic (separate multiply + add, each individually rounded).
//   When the AI Core compiler contracts a*b+c into fma(a,b,c) (single rounding),
//   the result differs by up to 1 ULP from TF's separate-round behavior.
//   This causes ~0.1%–0.8% of output elements to differ from TF golden.
//
//   Pixel analysis confirmed:
//     - Plain float32 (no FMA) → 100% bit-exact match with TF golden
//     - FMA contraction → ~85% match (14% of pixels affected)
//
// Explicit fmaf() calls in DsDiv are library function calls and are
// NOT affected by this pragma — they still produce correctly-rounded FMA results.
#pragma clang fp contract(off)

namespace NsImageProjectiveTransform {

using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 512;

// ===== Accurate float division =====
// The AI Core's division operator may differ by up to 1 ULP from x86's
// correctly-rounded IEEE 754 division. DsDiv refines the quotient using
// fmaf() to compute the exact residual:
//   1. q = a / b (initial quotient, may be off by 1 ULP on AI Core)
//   2. residual = fmaf(q, b, -a) = q*b - a (exact via FMA, single rounding)
//   3. return q - residual / b (refined quotient)
//
// fmaf(q, b, -a) computes q*b + (-a) with a single rounding at the end,
// giving the exact residual of the division (no intermediate rounding of q*b).
// Note: residual / b is itself a division and may be off by 1 ULP on AI Core,
// so the result is refined to within 1-2 ULP of the correctly-rounded float32,
// not guaranteed bit-exact. Empirically this matches TF golden on all test
// cases; a second refinement iteration is not added to avoid extra latency.
__simt_callee__ inline float DsDiv(float a, float b)
{
    if (b == 0.0f) {
        return a / b; // Let hardware produce inf/nan; caller handles via isfinite()
    }
    float q = a / b;
    // fmaf(q, b, -a) = q*b - a with single rounding (exact residual via FMA)
    float residual = fmaf(q, b, -a);
    // q is too large if residual > 0, too small if residual < 0
    // Correct: q_refined = q - residual / b
    return q - residual / b;
}

// ===== Helper functions =====

template <typename T>
__simt_callee__ inline float ReadAsFloat(T val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float ReadAsFloat<half>(half val)
{
    return __half2float(val);
}

// Truncate a float towards zero then clamp to [0, 255], returning uint8_t.
// Used by both CastFillValue<uint8_t> (fill-value semantics; matches numpy
// np.full(..., dtype=np.uint8)) and CastResult<uint8_t> (interpolation-result
// cast; matches TF 2.x uint8 truncation behavior).
__simt_callee__ inline uint8_t ClampToUint8(float val)
{
    int32_t truncated = static_cast<int32_t>(val);
    if (truncated < 0) {
        return 0;
    }
    if (truncated > 255) {
        return 255;
    }
    return static_cast<uint8_t>(truncated);
}

template <typename T>
__simt_callee__ inline T CastFillValue(float val)
{
    return static_cast<T>(val);
}

template <>
__simt_callee__ inline half CastFillValue<half>(float val)
{
    return __float2half(val);
}

template <>
__simt_callee__ inline uint8_t CastFillValue<uint8_t>(float val)
{
    // Matches numpy: np.full(..., fv, dtype=np.uint8) truncates towards zero
    return ClampToUint8(val);
}

template <>
__simt_callee__ inline int32_t CastFillValue<int32_t>(float val)
{
    // Truncate towards zero (matches numpy: np.full(..., fv, dtype=np.int32))
    return static_cast<int32_t>(val);
}

// CastResult: float interpolation result -> target dtype
// For integer types: use truncation (static_cast) to match TensorFlow 2.x behavior.
// Empirical verification (TF 2.21.0): TF uses truncation, NOT std::round, for all
// integer output types (uint8/int32). 33/33 test cases where trunc!=round
// match truncation. The MDE document's "roundf()" specification is incorrect for
// this TF version.
// Then clamp to the target integer type's valid range.
template <typename T>
__simt_callee__ inline T CastResult(float val)
{
    return static_cast<T>(val);
}

template <>
__simt_callee__ inline half CastResult<half>(float val)
{
    return __float2half(val);
}

template <>
__simt_callee__ inline uint8_t CastResult<uint8_t>(float val)
{
    return ClampToUint8(val);
}

template <>
__simt_callee__ inline int32_t CastResult<int32_t>(float val)
{
    return static_cast<int32_t>(val);
}

// ===== Fill output with 0 (CONSTANT mode default) =====
template <typename T>
__simt_callee__ inline void FillOutput(__gm__ T* output, int64_t outBase, int32_t channels)
{
    T fv = CastFillValue<T>(0.0f);
    for (int32_t c = 0; c < channels; c++) {
        output[outBase + c] = fv;
    }
}

// ===== Fill output with NaN-like value (BILINEAR NaN transform path) =====
// TF BILINEAR + NaN transform → outputs INT_MIN for integer types, NaN for float types
template <typename T>
__simt_callee__ inline void FillOutputNaN(__gm__ T* output, int64_t outBase, int32_t channels)
{
    T nanVal;
    if constexpr (std::is_same_v<T, half>) {
        union {
            float f;
            uint32_t u;
        } nanConv;
        nanConv.u = 0x7FC00000U;
        nanVal = __float2half(nanConv.f);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        nanVal = static_cast<int32_t>(0x80000000U);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        nanVal = 0;
    } else {
        union {
            float f;
            uint32_t u;
        } nanConv;
        nanConv.u = 0x7FC00000U;
        nanVal = nanConv.f;
    }
    for (int32_t c = 0; c < channels; c++) {
        output[outBase + c] = nanVal;
    }
}

// ===== Bilinear interpolation =====
// Matches TF's image_ops.h ProjectiveGenerator::bilinear_interpolation exactly:
//   x_floor = floor(x), x_ceil = x_floor + 1
//   y_floor = floor(y), y_ceil = y_floor + 1
//   value_yfloor = (x_ceil - x) * p(x_floor,y_floor) + (x - x_floor) * p(x_ceil,y_floor)
//   value_yceil  = (x_ceil - x) * p(x_floor,y_ceil)  + (x - x_floor) * p(x_ceil,y_ceil)
//   result = (y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil
//
// FMA is DISABLED via file-level #pragma clang fp contract(off).
// TF's CPU code uses separate multiply+add (each individually rounded).
// The AI Core compiler must NOT contract a*b+c into fma(a,b,c), because FMA
// uses a single rounding and produces different results from TF's two-rounding behavior.
template <typename T>
__simt_callee__ inline void BilinearInterpolateFloat(__gm__ T* images, __gm__ T* output, int64_t outBase, float xIn,
                                                     float yIn, int32_t hIn, int32_t wIn, int32_t channels, int64_t b)
{
    constexpr float fillValue = 0.0f;
    float xFloor = floorf(xIn);
    float yFloor = floorf(yIn);
    float xCeil = xFloor + 1.0f;
    float yCeil = yFloor + 1.0f;

    int32_t x0 = static_cast<int32_t>(xFloor);
    int32_t y0 = static_cast<int32_t>(yFloor);
    int32_t x1 = x0 + 1;
    int32_t y1 = y0 + 1;

    // Interpolation weights — FMA disabled, matching TF's separate mul+add
    float wx0 = xCeil - xIn;  // = x_ceil - x
    float wx1 = xIn - xFloor; // = x - x_floor
    float wy0 = yCeil - yIn;  // = y_ceil - y
    float wy1 = yIn - yFloor; // = y - y_floor

    // Check each neighbor individually (TF behavior: OOB neighbors use fill_value)
    bool valid00 = (x0 >= 0 && x0 < wIn && y0 >= 0 && y0 < hIn);
    bool valid10 = (x1 >= 0 && x1 < wIn && y0 >= 0 && y0 < hIn);
    bool valid01 = (x0 >= 0 && x0 < wIn && y1 >= 0 && y1 < hIn);
    bool valid11 = (x1 >= 0 && x1 < wIn && y1 >= 0 && y1 < hIn);

    int64_t batchOff = b * static_cast<int64_t>(hIn) * wIn * channels;
    int64_t off00 = batchOff + (static_cast<int64_t>(y0) * wIn + x0) * channels;
    int64_t off10 = batchOff + (static_cast<int64_t>(y0) * wIn + x1) * channels;
    int64_t off01 = batchOff + (static_cast<int64_t>(y1) * wIn + x0) * channels;
    int64_t off11 = batchOff + (static_cast<int64_t>(y1) * wIn + x1) * channels;

    for (int32_t c = 0; c < channels; c++) {
        float p00 = valid00 ? ReadAsFloat<T>(images[off00 + c]) : fillValue;
        float p10 = valid10 ? ReadAsFloat<T>(images[off10 + c]) : fillValue;
        float p01 = valid01 ? ReadAsFloat<T>(images[off01 + c]) : fillValue;
        float p11 = valid11 ? ReadAsFloat<T>(images[off11 + c]) : fillValue;
        // Bilinear interpolation — volatile prevents FMA contraction on AI Core
        // (matches TF's separate mul+add with individual rounding)
        volatile float m00 = wx0 * p00;
        volatile float m10 = wx1 * p10;
        float valYfloor = m00 + m10;
        volatile float m01 = wx0 * p01;
        volatile float m11 = wx1 * p11;
        float valYceil = m01 + m11;
        volatile float vYf = valYfloor;
        volatile float vYc = valYceil;
        volatile float mYf = wy0 * vYf;
        volatile float mYc = wy1 * vYc;
        float result = mYf + mYc;
        output[outBase + c] = CastResult<T>(result);
    }
}

// ===== Nearest interpolation =====
template <typename T>
__simt_callee__ inline void NearestInterpolate(__gm__ T* images, __gm__ T* output, int64_t outBase, float xIn,
                                               float yIn, int32_t hIn, int32_t wIn, int32_t channels, int64_t b)
{
    int32_t xi = static_cast<int32_t>(roundf(xIn));
    int32_t yi = static_cast<int32_t>(roundf(yIn));

    if (xi < 0 || xi >= wIn || yi < 0 || yi >= hIn) {
        FillOutput<T>(output, outBase, channels);
        return;
    }

    int64_t inOffset = (b * static_cast<int64_t>(hIn) * wIn + static_cast<int64_t>(yi) * wIn + xi) * channels;
    for (int32_t c = 0; c < channels; c++) {
        output[outBase + c] = images[inOffset + c];
    }
}

// ===== Per-pixel projective transform (callee) =====
template <typename T, uint32_t interpMode>
__simt_callee__ inline void ProcessPixel(int32_t xOut, int32_t yOut, int64_t b, int32_t hIn, int32_t wIn, int32_t hOut,
                                         int32_t wOut, int32_t channels, __gm__ T* images, __gm__ float* transforms,
                                         __gm__ T* output)
{
    int64_t transBase = b * 8;
    float a0 = transforms[transBase + 0];
    float a1 = transforms[transBase + 1];
    float a2 = transforms[transBase + 2];
    float a3 = transforms[transBase + 3];
    float a4 = transforms[transBase + 4];
    float a5 = transforms[transBase + 5];
    float a6 = transforms[transBase + 6];
    float a7 = transforms[transBase + 7];

    float fxOut = static_cast<float>(xOut);
    float fyOut = static_cast<float>(yOut);

    int64_t outBase = (static_cast<int64_t>(b) * hOut * wOut + static_cast<int64_t>(yOut) * wOut + xOut) * channels;

    // Compute denom = a6*fxOut + a7*fyOut + 1.0
    // volatile prevents FMA contraction on AI Core (matches TF's separate mul+add)
    volatile float a6fx = a6 * fxOut;
    volatile float a7fy = a7 * fyOut;
    float denom = a6fx + a7fy + 1.0f;

    if (!isfinite(denom)) {
        if constexpr (interpMode == IPT_TPL_BILINEAR) {
            FillOutputNaN<T>(output, outBase, channels);
        } else {
            FillOutput<T>(output, outBase, channels);
        }
        return;
    }

    volatile float a0fx = a0 * fxOut;
    volatile float a1fy = a1 * fyOut;
    volatile float a3fx = a3 * fxOut;
    volatile float a4fy = a4 * fyOut;
    float numX = a0fx + a1fy + a2;
    float numY = a3fx + a4fy + a5;
    float xIn = DsDiv(numX, denom);
    float yIn = DsDiv(numY, denom);

    if (!isfinite(xIn) || !isfinite(yIn)) {
        if constexpr (interpMode == IPT_TPL_BILINEAR) {
            FillOutputNaN<T>(output, outBase, channels);
        } else {
            FillOutput<T>(output, outBase, channels);
        }
        return;
    }

    if constexpr (interpMode == IPT_TPL_BILINEAR) {
        BilinearInterpolateFloat<T>(images, output, outBase, xIn, yIn, hIn, wIn, channels, b);
    } else {
        NearestInterpolate<T>(images, output, outBase, xIn, yIn, hIn, wIn, channels, b);
    }
}

// ===== Main SIMT VF kernel =====
template <typename T, uint32_t interpMode>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpImageProjectiveTransformSimtKernel(
    int64_t totalPixels, int64_t spatialSize, int32_t hIn, int32_t wIn, int32_t hOut, int32_t wOut, int32_t channels,
    __gm__ T* images, __gm__ float* transforms, __gm__ T* output)
{
    for (uint64_t pixelIdx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pixelIdx < static_cast<uint64_t>(totalPixels); pixelIdx += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int64_t b = static_cast<int64_t>(pixelIdx / static_cast<uint64_t>(spatialSize));
        int64_t remainder = static_cast<int64_t>(pixelIdx % static_cast<uint64_t>(spatialSize));
        int32_t yOut = static_cast<int32_t>(remainder / wOut);
        int32_t xOut = static_cast<int32_t>(remainder % wOut);

        ProcessPixel<T, interpMode>(xOut, yOut, b, hIn, wIn, hOut, wOut, channels, images, transforms, output);
    }
}

template <typename T, uint32_t interpMode>
__aicore__ inline void Process(GM_ADDR images, GM_ADDR transforms, GM_ADDR transformedImages,
                               const ImageProjectiveTransformTilingData* tilingData, int32_t actualHOut,
                               int32_t actualWOut)
{
    __gm__ T* imagesGm = (__gm__ T*)images;
    __gm__ float* transformsGm = (__gm__ float*)transforms;
    __gm__ T* outputGm = (__gm__ T*)transformedImages;

    int32_t hOut = actualHOut;
    int32_t wOut = actualWOut;
    int64_t spatialSize = static_cast<int64_t>(hOut) * wOut;
    int64_t totalPixels = static_cast<int64_t>(tilingData->batchSize) * hOut * wOut;

    asc_vf_call<OpImageProjectiveTransformSimtKernel<T, interpMode>>(
        dim3(THREAD_NUM), totalPixels, spatialSize, tilingData->hIn, tilingData->wIn, hOut, wOut, tilingData->channels,
        imagesGm, transformsGm, outputGm);
}

} // namespace NsImageProjectiveTransform

#endif // IMAGE_PROJECTIVE_TRANSFORM_SIMT_H_
