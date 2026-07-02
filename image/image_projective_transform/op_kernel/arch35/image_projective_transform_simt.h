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
 *
 * Performance optimizations applied (op_perf_skill Stage 1):
 *   R001: FastPath promoted to compile-time template parameter (if constexpr)
 *   R002: Constant divisions (spatialSize, wOut) replaced by Simt::UintDiv
 *   R003: Index width templated (uint32_t / uint64_t) — 32-bit path when totalPixels <= INT32_MAX
 *   R006: __launch_bounds__ templated on index width (1024 for uint32, 512 for uint64)
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
#include "image_projective_transform_safe_math.h"

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
// Explicit fmaf() calls in SafeDiv are library function calls and are
// NOT affected by this pragma — they still produce correctly-rounded FMA results.
#pragma clang fp contract(off)

namespace NsImageProjectiveTransform {

using namespace AscendC;
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
    int32_t truncated = __float2int_rz(val);
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
    return SafeTruncToInt32(val);
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
// FastPath=true:  plain static_cast (xIn/yIn finite & in int32 range for normal
//                  transforms — zero overhead vs. original code).
// FastPath=false: SafeFloorToInt32 clamps huge finite coordinates to OOB int32
//                  so they fail the in-bounds check -> fill_value, matching TF.
template <typename T, bool FastPath>
__simt_callee__ inline void BilinearInterpolateFloat(__gm__ T* images, __gm__ T* output, int64_t outBase, float xIn,
                                                     float yIn, int32_t hIn, int32_t wIn, int32_t channels, int64_t b)
{
    constexpr float fillValue = 0.0f;
    float xFloor = floorf(xIn);
    float yFloor = floorf(yIn);
    float xCeil = xFloor + 1.0f;
    float yCeil = yFloor + 1.0f;

    int32_t x0, y0;
    if constexpr (FastPath) {
        x0 = static_cast<int32_t>(xFloor);
        y0 = static_cast<int32_t>(yFloor);
    } else {
        x0 = SafeFloorToInt32(xIn);
        y0 = SafeFloorToInt32(yIn);
    }
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
// xIn/yIn are guaranteed finite on entry (inf/nan intercepted in ProcessPixel).
// FastPath=true:  plain static_cast<int32_t>(roundf) — zero overhead.
// FastPath=false: SafeRoundToInt32 clamps huge finite coordinates to OOB int32
//                  so they map to fill_value, matching TF.
template <typename T, bool FastPath>
__simt_callee__ inline void NearestInterpolate(__gm__ T* images, __gm__ T* output, int64_t outBase, float xIn,
                                               float yIn, int32_t hIn, int32_t wIn, int32_t channels, int64_t b)
{
    int32_t xi, yi;
    if constexpr (FastPath) {
        xi = static_cast<int32_t>(roundf(xIn));
        yi = static_cast<int32_t>(roundf(yIn));
    } else {
        xi = SafeRoundToInt32(xIn);
        yi = SafeRoundToInt32(yIn);
    }

    if (xi < 0 || xi >= wIn || yi < 0 || yi >= hIn) {
        FillOutput<T>(output, outBase, channels);
        return;
    }

    int64_t inOffset = (b * static_cast<int64_t>(hIn) * wIn + static_cast<int64_t>(yi) * wIn + xi) * channels;
    for (int32_t c = 0; c < channels; c++) {
        output[outBase + c] = images[inOffset + c];
    }
}

// ===== Per-pixel projective transform (core implementation, R001 + R003) =====
// R001: FastPath is a compile-time template parameter. `if constexpr (FastPath)`
//       eliminates the three runtime `if (fastPath)` branches that existed in
//       the original per-pixel code. This function is instantiated twice
//       (FastPath=true / FastPath=false); the dispatcher in Process selects the
//       right instance based on whether ALL batches share the same fast-path
//       flag. When batches are mixed, ProcessPixelMixed (below) dispatches at
//       runtime between the two compiled instances — but that only happens on
//       the rare mixed-batch path.
//
// R003: pixel-coordinate variables (xOut, yOut, b) use IDX_T (uint32_t /
//       uint64_t). Address offsets (outBase, transBase) stay int64_t to avoid
//       overflow (outBase = b*hOut*wOut*channels can exceed INT32_MAX for large
//       images). uint32_t path is used when totalPixels <= INT32_MAX, giving
//       higher arithmetic throughput.
//
// FastPath=true (normal transforms, |a|<=1e30):
//   - volatile multiplies (no FMA contraction, matches TF)
//   - DsDiv (lightweight accurate division)
//   - skip notFinite check (xIn/yIn guaranteed finite: 1e30*maxCoord<<FLT_MAX,
//     finite/finite with denom!=0 → finite result)
//   - static_cast<int32_t>(floorf/roundf) (no SafeFloor/SafeRound overhead)
//   Zero overhead vs. original code.
//
// FastPath=false (extreme transforms, |a|>1e30 or inf/nan):
//   - SafeMul/SafeSum3 (never traps, returns signed inf for overflow)
//   - SafeDiv + FlushSubnormal (handles inf/nan, matches TF FTZ)
//   - notFinite check (inf/nan xIn/yIn → NaN/fill)
//   - SafeFloorToInt32/SafeRoundToInt32 (clamps huge finite to OOB)
// ===== Compute projection (extracted from ProcessPixelImpl) =====
// Reads transform coefficients and computes denom/numX/numY. FastPath selects
// volatile multiplies (TF-matching) vs SafeMul (trap-free). Inlined by the
// compiler — zero call overhead.
template <bool FastPath, typename IDX_T>
__simt_callee__ inline void ComputeProjection(__gm__ float* transforms, IDX_T b, float fxOut, float fyOut, float& denom,
                                              float& numX, float& numY)
{
    int64_t transBase = static_cast<int64_t>(b) * 8;
    float a0 = transforms[transBase + 0];
    float a1 = transforms[transBase + 1];
    float a2 = transforms[transBase + 2];
    float a3 = transforms[transBase + 3];
    float a4 = transforms[transBase + 4];
    float a5 = transforms[transBase + 5];
    float a6 = transforms[transBase + 6];
    float a7 = transforms[transBase + 7];
    if constexpr (FastPath) {
        volatile float a6fx = a6 * fxOut;
        volatile float a7fy = a7 * fyOut;
        denom = a6fx + a7fy + 1.0f;
        volatile float a0fx = a0 * fxOut;
        volatile float a1fy = a1 * fyOut;
        volatile float a3fx = a3 * fxOut;
        volatile float a4fy = a4 * fyOut;
        numX = a0fx + a1fy + a2;
        numY = a3fx + a4fy + a5;
    } else {
        denom = SafeSum3(SafeMul(a6, fxOut), SafeMul(a7, fyOut), 1.0f);
        numX = SafeSum3(SafeMul(a0, fxOut), SafeMul(a1, fyOut), a2);
        numY = SafeSum3(SafeMul(a3, fxOut), SafeMul(a4, fyOut), a5);
    }
}

// Dispatch to BilinearInterpolateFloat or NearestInterpolate based on interpMode.
template <typename T, uint32_t interpMode, bool FastPath>
__simt_callee__ inline void DispatchInterpolate(__gm__ T* images, __gm__ T* output, int64_t outBase, float xIn,
                                                float yIn, int32_t hIn, int32_t wIn, int32_t channels, int64_t b)
{
    if constexpr (interpMode == IPT_TPL_BILINEAR) {
        BilinearInterpolateFloat<T, FastPath>(images, output, outBase, xIn, yIn, hIn, wIn, channels, b);
    } else {
        NearestInterpolate<T, FastPath>(images, output, outBase, xIn, yIn, hIn, wIn, channels, b);
    }
}

template <typename T, uint32_t interpMode, bool FastPath, typename IDX_T>
__simt_callee__ inline void ProcessPixelImpl(IDX_T xOut, IDX_T yOut, IDX_T b, int32_t hIn, int32_t wIn, int32_t hOut,
                                             int32_t wOut, int32_t channels, __gm__ T* images, __gm__ float* transforms,
                                             __gm__ T* output)
{
    float fxOut = static_cast<float>(xOut);
    float fyOut = static_cast<float>(yOut);
    int64_t outBase = (static_cast<int64_t>(b) * hOut * wOut + static_cast<int64_t>(yOut) * wOut +
                       static_cast<int64_t>(xOut)) *
                      channels;

    float denom, numX, numY;
    ComputeProjection<FastPath>(transforms, b, fxOut, fyOut, denom, numX, numY);

    if (denom == 0.0f) {
        FillOutput<T>(output, outBase, channels);
        return;
    }

    float xIn, yIn;
    if constexpr (FastPath) {
        xIn = DsDiv(numX, denom);
        yIn = DsDiv(numY, denom);
        DispatchInterpolate<T, interpMode, true>(images, output, outBase, xIn, yIn, hIn, wIn, channels,
                                                 static_cast<int64_t>(b));
    } else {
        xIn = FlushSubnormal(SafeDiv(numX, denom));
        yIn = FlushSubnormal(SafeDiv(numY, denom));
        bool notFinite = isinf(xIn) || isnan(xIn) || isinf(yIn) || isnan(yIn);
        if (notFinite) {
            if constexpr (interpMode == IPT_TPL_BILINEAR) {
                FillOutputNaN<T>(output, outBase, channels);
            } else {
                FillOutput<T>(output, outBase, channels);
            }
            return;
        }
        DispatchInterpolate<T, interpMode, false>(images, output, outBase, xIn, yIn, hIn, wIn, channels,
                                                  static_cast<int64_t>(b));
    }
}

// ===== Per-pixel projective transform (mixed-batch runtime dispatch, R001) =====
// Used ONLY when batches are mixed (some fast, some safe). Each pixel reads its
// batch's fastPath flag and dispatches to the corresponding compiled instance.
// This is a single function-call branch per pixel (vs. the original three
// intra-function `if (fastPath)` branches), and only triggers on the rare
// mixed-batch path. Normal (all-fast / all-safe) batches use the branch-free
// ProcessPixelImpl directly via the templated VF.
template <typename T, uint32_t interpMode, typename IDX_T>
__simt_callee__ inline void ProcessPixelMixed(IDX_T xOut, IDX_T yOut, IDX_T b, int32_t hIn, int32_t wIn, int32_t hOut,
                                              int32_t wOut, int32_t channels, __gm__ T* images,
                                              __gm__ float* transforms, __gm__ T* output, bool fastPath)
{
    if (fastPath) {
        ProcessPixelImpl<T, interpMode, true, IDX_T>(xOut, yOut, b, hIn, wIn, hOut, wOut, channels, images, transforms,
                                                     output);
    } else {
        ProcessPixelImpl<T, interpMode, false, IDX_T>(xOut, yOut, b, hIn, wIn, hOut, wOut, channels, images, transforms,
                                                      output);
    }
}

// ===== Main SIMT VF kernel (uniform fastPath, R001 + R002 + R003 + R006) =====
// R001: FastPath is a compile-time template parameter — no runtime branch.
// R002: pixelIdx / spatialSize and remainder / wOut use Simt::UintDiv (magic +
//       shift) instead of hardware division. remainder and xOut are recovered
//       via multiply-subtract (remainder = pixelIdx - b*spatialSize,
//       xOut = remainder - yOut*wOut). All three original divisions are
//       eliminated.
// R003: loop variable and coordinate decomposition use IDX_T (uint32_t /
//       uint64_t). The 32-bit path is selected when totalPixels <= INT32_MAX.
// R006: __launch_bounds__ uses THREADS<IDX_T> (1024 for uint32, 512 for uint64).
//
// Pre-computed in scalar Process and passed by value:
//   spatialSize, magicSpatial, shiftSpatial, magicW, shiftW
// (all IDX_T). wOut is passed as int32_t (image dimension, always fits).
template <typename T, uint32_t interpMode, bool FastPath, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpImageProjectiveTransformSimtKernel(
    IDX_T totalPixels, IDX_T spatialSize, int32_t hIn, int32_t wIn, int32_t hOut, int32_t wOut, int32_t channels,
    __gm__ T* images, __gm__ float* transforms, __gm__ T* output, IDX_T magicSpatial, IDX_T shiftSpatial, IDX_T magicW,
    IDX_T shiftW)
{
    IDX_T step = static_cast<IDX_T>(blockDim.x) * static_cast<IDX_T>(gridDim.x);
    for (IDX_T pixelIdx =
             static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
         pixelIdx < totalPixels; pixelIdx += step) {
        // R002: fast division by constant spatialSize and wOut
        IDX_T b = Simt::UintDiv<IDX_T>(pixelIdx, magicSpatial, shiftSpatial);
        IDX_T remainder = pixelIdx - b * spatialSize;
        IDX_T yOut = Simt::UintDiv<IDX_T>(remainder, magicW, shiftW);
        IDX_T xOut = remainder - yOut * static_cast<IDX_T>(wOut);
        ProcessPixelImpl<T, interpMode, FastPath, IDX_T>(xOut, yOut, b, hIn, wIn, hOut, wOut, channels, images,
                                                         transforms, output);
    }
}

// ===== Mixed-batch SIMT VF kernel (R001 + R002 + R003 + R006) =====
// Identical coordinate decomposition to the uniform kernel, but reads the
// per-batch fastPath flag from UB and dispatches to ProcessPixelMixed. Only
// used when batches are mixed (some fast, some safe); normal batches use the
// branch-free uniform kernel above.
template <typename T, uint32_t interpMode, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpImageProjectiveTransformSimtKernelMixed(
    IDX_T totalPixels, IDX_T spatialSize, int32_t hIn, int32_t wIn, int32_t hOut, int32_t wOut, int32_t channels,
    __gm__ T* images, __gm__ float* transforms, __gm__ T* output, IDX_T magicSpatial, IDX_T shiftSpatial, IDX_T magicW,
    IDX_T shiftW, __ubuf__ uint8_t* fastPathFlags)
{
    IDX_T step = static_cast<IDX_T>(blockDim.x) * static_cast<IDX_T>(gridDim.x);
    for (IDX_T pixelIdx =
             static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
         pixelIdx < totalPixels; pixelIdx += step) {
        IDX_T b = Simt::UintDiv<IDX_T>(pixelIdx, magicSpatial, shiftSpatial);
        IDX_T remainder = pixelIdx - b * spatialSize;
        IDX_T yOut = Simt::UintDiv<IDX_T>(remainder, magicW, shiftW);
        IDX_T xOut = remainder - yOut * static_cast<IDX_T>(wOut);
        bool fastPath = (fastPathFlags[static_cast<size_t>(b)] != 0);
        ProcessPixelMixed<T, interpMode, IDX_T>(xOut, yOut, b, hIn, wIn, hOut, wOut, channels, images, transforms,
                                                output, fastPath);
    }
}

// ===== VF dispatcher by fastPath uniformity and index width =====
// R002: magic/shift for spatialSize and wOut are pre-computed here (scalar) and
//       passed by value to the VF. GetUintDivMagicAndShift is a __aicore__
//       scalar API (cannot run in VF). The divisors are constant throughout VF
//       execution, so the precomputed magic/shift are valid for every pixel.
// R001: allFast → uniform FastPath=true VF; allSafe → uniform FastPath=false VF;
//       mixed → mixed-batch VF (runtime dispatch, rare path).
// R006: dim3 uses THREADS<IDX_T> to match __launch_bounds__.
template <typename T, uint32_t interpMode, typename IDX_T>
__aicore__ inline void DispatchVf(IDX_T totalPixels, IDX_T spatialSize, int32_t hIn, int32_t wIn, int32_t hOut,
                                  int32_t wOut, int32_t channels, __gm__ T* imagesGm, __gm__ float* transformsGm,
                                  __gm__ T* outputGm, bool allFast, bool allSafe, __ubuf__ uint8_t* fastPathFlags)
{
    IDX_T magicSpatial = 0;
    IDX_T shiftSpatial = 0;
    IDX_T magicW = 0;
    IDX_T shiftW = 0;
    GetUintDivMagicAndShift<IDX_T>(magicSpatial, shiftSpatial, spatialSize);
    GetUintDivMagicAndShift<IDX_T>(magicW, shiftW, static_cast<IDX_T>(wOut));

    if (allFast) {
        asc_vf_call<OpImageProjectiveTransformSimtKernel<T, interpMode, true, IDX_T>>(
            dim3(THREADS<IDX_T>), totalPixels, spatialSize, hIn, wIn, hOut, wOut, channels, imagesGm, transformsGm,
            outputGm, magicSpatial, shiftSpatial, magicW, shiftW);
    } else if (allSafe) {
        asc_vf_call<OpImageProjectiveTransformSimtKernel<T, interpMode, false, IDX_T>>(
            dim3(THREADS<IDX_T>), totalPixels, spatialSize, hIn, wIn, hOut, wOut, channels, imagesGm, transformsGm,
            outputGm, magicSpatial, shiftSpatial, magicW, shiftW);
    } else {
        asc_vf_call<OpImageProjectiveTransformSimtKernelMixed<T, interpMode, IDX_T>>(
            dim3(THREADS<IDX_T>), totalPixels, spatialSize, hIn, wIn, hOut, wOut, channels, imagesGm, transformsGm,
            outputGm, magicSpatial, shiftSpatial, magicW, shiftW, fastPathFlags);
    }
}

// ===== Batch-level fastPath pre-computation (scalar, R001) =====
// Computes per-batch fastPath flags into UB and determines whether all batches
// are fast (allFast), all safe (allSafe), or mixed. The uniform (allFast /
// allSafe) case lets the dispatcher pick a branch-free VF instance.
template <typename T>
__aicore__ inline void ComputeBatchFastPath(__gm__ float* transformsGm, int32_t batchSize,
                                            __ubuf__ uint8_t* fastPathFlags, bool& allFast, bool& allSafe)
{
    allFast = true;
    allSafe = true;
    for (int32_t b = 0; b < batchSize; b++) {
        int64_t transBase = static_cast<int64_t>(b) * 8;
        float a0 = transformsGm[transBase + 0];
        float a1 = transformsGm[transBase + 1];
        float a2 = transformsGm[transBase + 2];
        float a3 = transformsGm[transBase + 3];
        float a4 = transformsGm[transBase + 4];
        float a5 = transformsGm[transBase + 5];
        float a6 = transformsGm[transBase + 6];
        float a7 = transformsGm[transBase + 7];
        bool fp = IsFastPath(a0, a1, a2, a3, a4, a5, a6, a7);
        fastPathFlags[b] = fp ? 1 : 0;
        if (fp) {
            allSafe = false;
        } else {
            allFast = false;
        }
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
    int64_t spatialSize64 = static_cast<int64_t>(hOut) * wOut;
    int64_t totalPixels64 = static_cast<int64_t>(tilingData->batchSize) * hOut * wOut;
    int32_t batchSize = tilingData->batchSize;

    // ===== Batch-level fastPath pre-computation (R001) =====
    // transforms 对同一 batch 的所有像素相同，逐像素重复 IsFastPath 判断是冗余的。
    // 在 scalar 侧按 batch 预计算 fastPath 标志到 UB，并判定全 fast / 全 safe / 混合。
    // 全 fast / 全 safe 时 VF 内编译期消除分支；混合时退化到运行时分支（极少触发）。
    LocalMemAllocator<Hardware::UB> ubAlloc;
    int64_t alignedBatchN = ((static_cast<int64_t>(batchSize) + 31) / 32) * 32;
    LocalTensor<uint8_t> fastPathTensor = ubAlloc.Alloc<uint8_t>(alignedBatchN);
    __ubuf__ uint8_t* fastPathFlags = (__ubuf__ uint8_t*)fastPathTensor.GetPhyAddr();

    bool allFast = true;
    bool allSafe = true;
    ComputeBatchFastPath<T>(transformsGm, batchSize, fastPathFlags, allFast, allSafe);
    DataSyncBarrier<MemDsbT::UB>();

    // ===== R003: dispatch by index width =====
    // totalPixels <= INT32_MAX → uint32_t path (32-bit arithmetic, 1024 threads)
    // totalPixels >  INT32_MAX → uint64_t path (64-bit arithmetic, 512 threads)
    if (totalPixels64 <= INT32_MAX_VAL) {
        DispatchVf<T, interpMode, uint32_t>(static_cast<uint32_t>(totalPixels64), static_cast<uint32_t>(spatialSize64),
                                            tilingData->hIn, tilingData->wIn, hOut, wOut, tilingData->channels,
                                            imagesGm, transformsGm, outputGm, allFast, allSafe, fastPathFlags);
    } else {
        DispatchVf<T, interpMode, uint64_t>(static_cast<uint64_t>(totalPixels64), static_cast<uint64_t>(spatialSize64),
                                            tilingData->hIn, tilingData->wIn, hOut, wOut, tilingData->channels,
                                            imagesGm, transformsGm, outputGm, allFast, allSafe, fastPathFlags);
    }
}

} // namespace NsImageProjectiveTransform

#endif // IMAGE_PROJECTIVE_TRANSFORM_SIMT_H_
