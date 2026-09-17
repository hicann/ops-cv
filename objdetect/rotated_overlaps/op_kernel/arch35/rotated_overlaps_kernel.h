/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file rotated_overlaps_kernel.h
 * \brief Pair-parallel SIMT and vector fallback implementations of RotatedOverlaps.
 *
 * The SIMT path distributes flattened [b, n, m] pairs with a grid-stride loop.
 * Each thread handles a pair: validity/axis-aligned/disjoint fast paths precede
 * smaller-first compensated polygon clipping and area accumulation.
 *
 * The vector fallback distributes contiguous tile-task ranges to cores, not
 * complete rows. A task vectorizes queries (contiguous output) or boxes
 * (strided output), broadcasting the other box. Each output has one writer.
 * Only this fallback constructs the fixed candidate set per vector lane:
 *
 *   - four A vertices inside B;
 *   - four B vertices inside A;
 *   - sixteen pairwise edge intersections.
 *
 * The 24 candidates plus eight invalid padding slots are sorted in-place with
 * a fixed 32-way bitonic network using a pseudo-angle key.  Invalid entries
 * are replaced with the first sorted point before the vector shoelace sum.
 * This gives a fixed-control-flow polygon area calculation without scalar
 * compaction, per-pair lists, or atomic output updates.
 */

#ifndef ROTATED_OVERLAPS_KERNEL_H_
#define ROTATED_OVERLAPS_KERNEL_H_

#include <cstdint>

#include "kernel_operator.h"
#include "lib/math/sincos.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"

#include "rotated_overlaps_tiling_data.h"
#include "rotated_overlaps_tiling_key.h"

namespace NsRotatedOverlaps {

using namespace AscendC;

constexpr uint32_t kCoordinateCount = 5U;
constexpr uint32_t kCornerCount = 4U;
constexpr uint32_t kRealCandidateCount = 24U;
constexpr uint32_t kCandidateCount = 32U;
constexpr uint32_t kAlignElements = 8U; // float32 32 B datablock
constexpr uint32_t kMaskCount = 3U;
constexpr uint32_t kMaskStrideBytes = 32U;

// A split representation keeps the float32 angle multiplication close to the
// correctly-rounded value of degrees * pi / 180.  This matters for very small
// rotated intersections, where one ulp of angular error can dominate the area.
constexpr float kDegreesToRadians = 0.01745329238474369049F;
constexpr float kDegreesToRadiansLow = 1.3519960498364902e-10F;
constexpr float kHalf = 0.5F;
constexpr float kMaxFinite = 3.402823466e38F;
constexpr float kInvalidKey = 10.0F;
// Geometric margins, not area tolerances. With round-to-nearest float32 u=2^-24,
// 2^-19=32u reserves headroom over a first-order 16u budget for endpoint
// centers/extents, bound accumulation and comparison arithmetic. Magnitude-sum
// scales avoid cancellation; the local-frame bound retains endpoint low parts
// until final float evaluation. This budget assumes finite, normal arithmetic,
// not a universal bound for overflowing or subnormal intermediates.
// The looser 2^-10 margin includes coarse trigonometry/conversion error under
// the assumptions in DefinitelyDisjoint. The 3600-degree cutoff limits that
// budget; it is not an input validity limit. Only strict padded bounds classify
// pairs; non-finite scales or subnormal margins cannot establish rejection.
constexpr float kBoundPaddingFactor = 0.0000019073486328125F; // 2^-19 = 32 * float32 u
constexpr float kCoarsePaddingFactor = 0.0009765625F;         // 2^-10
constexpr float kCoarseAngleLimit = 3600.0F;                  // relative angle is bounded by 7200 degrees
constexpr float kMinNormal = 1.1754943508222875e-38F;

enum FloatVectorSlot : uint32_t {
    kQx = 0U,
    kQy = 1U,
    kQw = 2U,
    kQh = 3U,
    kQt = 4U,
    kAx = 5U,
    kAy = 6U,
    kAw = 7U,
    kAh = 8U,
    kAt = 9U,
    kSinA = 10U,
    kCosA = 11U,
    kSinQ = 12U,
    kCosQ = 13U,
    kAValid = 14U,
    kQValid = 15U,
    kPairValid = 16U,
    kOne = 17U,
    kCenterX = 18U,
    kCenterY = 19U,
    kTmp0 = 20U,
    kTmp1 = 21U,
    kTmp2 = 22U,
    kTmp3 = 23U,
    kTmp4 = 24U,
    kTmp5 = 25U,
    kTmp6 = 26U,
    kTmp7 = 27U,
    kACornerBase = 28U,
    kBCornerBase = 36U,
    kCandidateXBase = 44U,
    kCandidateYBase = 76U,
    kCandidateKeyBase = 108U,
    kSwapKey = 140U,
    kSwapX = 141U,
    kSwapY = 142U,
    kOutput = 143U,
    // kAt is converted from degrees to radians while computing a tile.  Keep
    // the broadcast box angle intact because one output row can span many
    // query tiles.
    kARawTheta = 144U,
    // CopyOutputStrided expands one result float to each 32-byte data block.
    // Keep its eight-vector destination separate from all live geometry data.
    kScatterScratchBase = 145U,
};

constexpr uint32_t kScatterScratchVectorCount = kAlignElements;
static_assert(kScatterScratchBase + kScatterScratchVectorCount == kRotatedOverlapsFloatVectorCount,
              "The strided-copy scratch region must be fully covered by the vector UB allocation.");

template <bool Use32Bit>
struct IndexTypeSelector {
    using type = uint64_t;
};

template <>
struct IndexTypeSelector<true> {
    using type = uint32_t;
};

struct PairBox {
    float centerX;
    float centerY;
    float width;
    float height;
    float theta;
    float lowerX;
    float lowerY;
    float upperX;
    float upperY;
    bool valid;
};

struct FloatExpansion {
    float high;
    float low;
};

template <bool Trans>
__simt_callee__ inline void LoadPairBox(__gm__ float* source, uint64_t sourceLength, uint64_t batch,
                                        uint64_t sourceIndex, PairBox& box)
{
    const uint64_t batchBase = batch * kCoordinateCount * sourceLength;
    const float rawX = source[batchBase + sourceIndex];
    const float rawY = source[batchBase + sourceLength + sourceIndex];
    const float rawW = source[batchBase + 2U * sourceLength + sourceIndex];
    const float rawH = source[batchBase + 3U * sourceLength + sourceIndex];
    box.theta = source[batchBase + 4U * sourceLength + sourceIndex];
    box.valid = isfinite(rawX) && isfinite(rawY) && isfinite(rawW) && isfinite(rawH) && isfinite(box.theta);
    if constexpr (Trans) {
        box.width = rawW - rawX;
        box.height = rawH - rawY;
        box.centerX = (rawX + rawW) * kHalf;
        box.centerY = (rawY + rawH) * kHalf;
        box.lowerX = rawX;
        box.lowerY = rawY;
        box.upperX = rawW;
        box.upperY = rawH;
    } else {
        box.centerX = rawX;
        box.centerY = rawY;
        box.width = rawW;
        box.height = rawH;
        box.lowerX = 0.0F;
        box.lowerY = 0.0F;
        box.upperX = 0.0F;
        box.upperY = 0.0F;
    }
    box.valid = box.valid && isfinite(box.centerX) && isfinite(box.centerY) && isfinite(box.width) &&
                isfinite(box.height) && box.width > 0.0F && box.height > 0.0F;
}

__simt_callee__ inline void TwoSum(float first, float second, float& sum, float& error)
{
    sum = first + second;
    const float secondVirtual = sum - first;
    const float firstVirtual = sum - secondVirtual;
    const float secondError = second - secondVirtual;
    const float firstError = first - firstVirtual;
    error = firstError + secondError;
}

__simt_callee__ inline FloatExpansion RenormalizeExpansion(float high, float low)
{
    FloatExpansion result;
    TwoSum(high, low, result.high, result.low);
    return result;
}

__simt_callee__ inline FloatExpansion AddExpansions(const FloatExpansion& first, const FloatExpansion& second)
{
    float high = 0.0F;
    float error = 0.0F;
    TwoSum(first.high, second.high, high, error);
    return RenormalizeExpansion(high, error + (first.low + second.low));
}

__simt_callee__ inline FloatExpansion NegateExpansion(const FloatExpansion& value) { return {-value.high, -value.low}; }

__simt_callee__ inline FloatExpansion SubtractExpansions(const FloatExpansion& first, const FloatExpansion& second)
{
    return AddExpansions(first, NegateExpansion(second));
}

__simt_callee__ inline FloatExpansion MultiplyExpansion(const FloatExpansion& value, float factor)
{
    const float product = value.high * factor;
    const float productError = fmaf(value.high, factor, -product);
    return RenormalizeExpansion(product, productError + value.low * factor);
}

__simt_callee__ inline float EvaluateExpansion(const FloatExpansion& value) { return value.high + value.low; }

__simt_callee__ inline FloatExpansion EndpointCenter(float lower, float upper)
{
    float sum = 0.0F;
    float error = 0.0F;
    TwoSum(lower, upper, sum, error);
    return {sum * kHalf, error * kHalf};
}

__simt_callee__ inline FloatExpansion EndpointExtent(float lower, float upper)
{
    FloatExpansion result;
    TwoSum(upper, -lower, result.high, result.low);
    return result;
}

__simt_callee__ inline bool ExpansionLess(float firstHigh, float firstLow, float secondHigh, float secondLow)
{
    return firstHigh < secondHigh || (firstHigh == secondHigh && firstLow < secondLow);
}

__simt_callee__ inline float ExactCenteredAxisOverlap(float firstCenter, float firstExtent, float secondCenter,
                                                      float secondExtent)
{
    float firstLowerHigh = 0.0F;
    float firstLowerLow = 0.0F;
    float firstUpperHigh = 0.0F;
    float firstUpperLow = 0.0F;
    float secondLowerHigh = 0.0F;
    float secondLowerLow = 0.0F;
    float secondUpperHigh = 0.0F;
    float secondUpperLow = 0.0F;
    const float firstHalfExtent = firstExtent * kHalf;
    const float secondHalfExtent = secondExtent * kHalf;
    TwoSum(firstCenter, -firstHalfExtent, firstLowerHigh, firstLowerLow);
    TwoSum(firstCenter, firstHalfExtent, firstUpperHigh, firstUpperLow);
    TwoSum(secondCenter, -secondHalfExtent, secondLowerHigh, secondLowerLow);
    TwoSum(secondCenter, secondHalfExtent, secondUpperHigh, secondUpperLow);

    float lowerHigh = firstLowerHigh;
    float lowerLow = firstLowerLow;
    if (ExpansionLess(firstLowerHigh, firstLowerLow, secondLowerHigh, secondLowerLow)) {
        lowerHigh = secondLowerHigh;
        lowerLow = secondLowerLow;
    }

    float upperHigh = firstUpperHigh;
    float upperLow = firstUpperLow;
    if (ExpansionLess(secondUpperHigh, secondUpperLow, firstUpperHigh, firstUpperLow)) {
        upperHigh = secondUpperHigh;
        upperLow = secondUpperLow;
    }

    float overlapHigh = 0.0F;
    float overlapLow = 0.0F;
    TwoSum(upperHigh, -lowerHigh, overlapHigh, overlapLow);
    const float overlap = overlapHigh + ((overlapLow + upperLow) - lowerLow);
    return overlap > 0.0F ? overlap : 0.0F;
}

template <bool Trans>
__simt_callee__ inline float AxisAlignedIntersectionArea(const PairBox& first, const PairBox& second)
{
    float overlapX = 0.0F;
    float overlapY = 0.0F;
    if constexpr (Trans) {
        const float lowerX = fmaxf(first.lowerX, second.lowerX);
        const float lowerY = fmaxf(first.lowerY, second.lowerY);
        const float upperX = fminf(first.upperX, second.upperX);
        const float upperY = fminf(first.upperY, second.upperY);
        overlapX = upperX > lowerX ? upperX - lowerX : 0.0F;
        overlapY = upperY > lowerY ? upperY - lowerY : 0.0F;
    } else {
        // Endpoint expansions preserve the low parts of center +/- halfExtent.
        // Taking their compensated min/max difference handles containment and
        // distinguishes exact contact from a positive one-ULP sliver without
        // an area-proportional zero threshold.
        overlapX = ExactCenteredAxisOverlap(first.centerX, first.width, second.centerX, second.width);
        overlapY = ExactCenteredAxisOverlap(first.centerY, first.height, second.centerY, second.height);
    }
    return overlapX > 0.0F && overlapY > 0.0F ? overlapX * overlapY : 0.0F;
}

__simt_callee__ inline FloatExpansion MultiplyExpansions(const FloatExpansion& a, const FloatExpansion& b)
{
    const float high = a.high * b.high;
    const float low = fmaf(a.high, b.high, -high) + (a.high * b.low + a.low * b.high);
    return RenormalizeExpansion(high, low);
}

__simt_callee__ inline FloatExpansion DivideExpansions(const FloatExpansion& a, const FloatExpansion& b)
{
    const float quotient = a.high / b.high;
    const FloatExpansion residual = SubtractExpansions(a, MultiplyExpansion(b, quotient));
    return RenormalizeExpansion(quotient, (residual.high + residual.low) / b.high);
}

__simt_callee__ inline void ExpandedSinCos(float degrees, FloatExpansion& sine, FloatExpansion& cosine)
{
    // Reduce in degrees before conversion; preserve the radian conversion
    // residual and evaluate on [-pi/4, pi/4] with compensated arithmetic.
    float reduced = fabsf(degrees);
    float period = 360.0F;
    while (period <= reduced * 0.5F) {
        period *= 2.0F;
    }
    while (period >= 360.0F) {
        if (reduced >= period) {
            reduced -= period;
        }
        period *= 0.5F;
    }
    if (degrees < 0.0F) {
        reduced = -reduced;
    }
    const int quadrant = static_cast<int>(floorf(reduced / 90.0F + 0.5F));
    const float remainder = reduced - static_cast<float>(quadrant) * 90.0F;
    const FloatExpansion radians = MultiplyExpansion({kDegreesToRadians, kDegreesToRadiansLow}, remainder);
    // Split Taylor coefficients preserve the low part without performing
    // compensated division for every term of every box pair. For k=1..8,
    // sin: (-1)^k/(2k+1)!, cos: (-1)^k/(2k)!, Horner in radians squared.
    // Generate c by binary64 division of the signed unit by the integer
    // factorial, then high=RN32(c), low=RN32(c-double(high)); RN32 is float32
    // round-to-nearest, ties-to-even. The subtraction is performed in binary64.
    // On |x|<=pi/4, alternating-series truncation is bounded by the next term:
    // sin: (pi/4)^19/19! ~= 8.35e-20; cos: (pi/4)^18/18! ~= 2.02e-18.
    // These are truncation bounds only; coefficient, conversion and Horner
    // rounding errors are separate, including the initial binary64 rounding.
    constexpr FloatExpansion sinCoefficients[8] = {
        {-0.1666666716337204F, 4.9670538793122887e-09F},      {0.0083333337679505348F, -4.3461720333759502e-10F},
        {-0.00019841270113829523F, 2.7255968749334558e-12F},  {2.7557318844628753e-06F, 3.7935712242972291e-14F},
        {-2.5052107943679403e-08F, -4.4176230446483665e-16F}, {1.6059044372074283e-10F, -5.3525265115627256e-18F},
        {-7.6471636098127127e-13F, -1.2200710471178288e-20F}, {2.8114573589663704e-15F, -1.0462084739763658e-22F}};
    constexpr FloatExpansion cosCoefficients[8] = {{-0.5F, 0.0F},
                                                   {0.041666667908430099F, -1.2417634698280722e-09F},
                                                   {-0.0013888889225199819F, 3.3631094437103215e-11F},
                                                   {2.4801587642286904e-05F, -3.4069960936668198e-13F},
                                                   {-2.755731998149713e-07F, 7.5751122090511949e-15F},
                                                   {2.0876755879584152e-09F, 1.1082839809204342e-16F},
                                                   {-1.147074536050896e-11F, -2.3722076892312381e-19F},
                                                   {4.7794772561329454e-14F, 7.6254440444864298e-22F}};
    const FloatExpansion square = MultiplyExpansions(radians, radians);
    FloatExpansion sinPolynomial = sinCoefficients[7];
    FloatExpansion cosPolynomial = cosCoefficients[7];
#pragma unroll
    for (int term = 6; term >= 0; --term) {
        sinPolynomial = AddExpansions(MultiplyExpansions(sinPolynomial, square), sinCoefficients[term]);
        cosPolynomial = AddExpansions(MultiplyExpansions(cosPolynomial, square), cosCoefficients[term]);
    }
    sine = MultiplyExpansions(radians, AddExpansions({1.0F, 0.0F}, MultiplyExpansions(square, sinPolynomial)));
    cosine = AddExpansions({1.0F, 0.0F}, MultiplyExpansions(square, cosPolynomial));
    const FloatExpansion sinReduced = sine;
    const FloatExpansion cosReduced = cosine;
    const int turn = (quadrant % 4 + 4) % 4;
    if (turn == 1) {
        sine = cosReduced;
        cosine = NegateExpansion(sinReduced);
    } else if (turn == 2) {
        sine = NegateExpansion(sinReduced);
        cosine = NegateExpansion(cosReduced);
    } else if (turn == 3) {
        sine = NegateExpansion(cosReduced);
        cosine = sinReduced;
    }
}

struct ExpandedPoint {
    FloatExpansion x;
    FloatExpansion y;
};

// Clipping a convex quadrilateral by four half-planes adds at most four
// vertices. Eight slots suffice even at intermediate stages. Keeping the
// high/low fields in flat arrays keeps the two polygon buffers at 256 B per
// thread, avoiding the stack overflow from oversized point arrays.
constexpr uint32_t kIntersectionCapacity = 8U;

__simt_callee__ inline ExpandedPoint LoadExpandedPoint(const float* coordinates, uint32_t index)
{
    return {{coordinates[index], coordinates[kIntersectionCapacity + index]},
            {coordinates[2U * kIntersectionCapacity + index], coordinates[3U * kIntersectionCapacity + index]}};
}

__simt_callee__ inline void StoreExpandedPoint(float* coordinates, uint32_t index, const ExpandedPoint& point)
{
    coordinates[index] = point.x.high;
    coordinates[kIntersectionCapacity + index] = point.x.low;
    coordinates[2U * kIntersectionCapacity + index] = point.y.high;
    coordinates[3U * kIntersectionCapacity + index] = point.y.low;
}

__simt_callee__ inline bool ExpansionNonnegative(const FloatExpansion& value)
{
    return value.high > 0.0F || (value.high == 0.0F && value.low >= 0.0F);
}

__simt_callee__ inline bool DefinitelyDisjoint(const PairBox& first, const PairBox& second)
{
    // Every rotation fits inside a square of half extent (width + height)/2.
    // Inflate for rounded endpoint centers/extents as well as this bound's
    // arithmetic. Overflow/underflow and near-contact fall through to clipping.
    const float radius = 0.5F * (first.width + first.height + second.width + second.height);
    const float scale = fabsf(first.centerX) + fabsf(first.centerY) + fabsf(second.centerX) + fabsf(second.centerY) +
                        radius;
    const float padding = scale * kBoundPaddingFactor;
    const float limit = radius + padding;
    if (isfinite(limit) && padding >= kMinNormal &&
        (fabsf(first.centerX - second.centerX) > limit || fabsf(first.centerY - second.centerY) > limit)) {
        return true;
    }
    // Rejection only, never an area approximation. Individual angles within
    // +/-3600 degrees give relative angles within +/-7200 (40*pi radians).
    // With u=2^-24, subtraction and degree conversion contribute approximately
    // 3*u*40*pi < 2.25e-5 radians. The budget assumes sincosf absolute error
    // <=2^-16 per component on this range; this is a design assumption, not a
    // vendor API guarantee. Conversion plus sincosf error is then <3.8e-5;
    // four projection contributions plus 32u arithmetic allowance total
    // <1.6e-4*scale, below 2^-10*scale. Revalidate for new math implementations.
    // Outside the angle range, with a non-finite scale/subnormal margin, or
    // near contact, this bound cannot safely decide: return false so the caller
    // uses compensated geometry (including its precise local-frame bounds).
    if (!isfinite(scale) || fabsf(first.theta) > kCoarseAngleLimit || fabsf(second.theta) > kCoarseAngleLimit) {
        return false;
    }
    const float coarsePadding = scale * kCoarsePaddingFactor;
    if (coarsePadding < kMinNormal) {
        return false;
    }
    float clipSin, clipCos, relativeSin, relativeCos;
    sincosf(second.theta * kDegreesToRadians, &clipSin, &clipCos);
    sincosf((first.theta - second.theta) * kDegreesToRadians, &relativeSin, &relativeCos);
    const float dx = first.centerX - second.centerX;
    const float dy = first.centerY - second.centerY;
    const float alignedX = fmaf(dy, clipSin, dx * clipCos);
    const float alignedY = fmaf(-dx, clipSin, dy * clipCos);
    const float extentX = 0.5F * (fabsf(relativeCos) * first.width + fabsf(relativeSin) * first.height);
    const float extentY = 0.5F * (fabsf(relativeSin) * first.width + fabsf(relativeCos) * first.height);
    return fabsf(alignedX) > extentX + 0.5F * second.width + coarsePadding ||
           fabsf(alignedY) > extentY + 0.5F * second.height + coarsePadding;
}

template <bool Trans>
__simt_callee__ inline float PreciseIntersectionArea(const PairBox& subject, const PairBox& clip)
{
    FloatExpansion subjectSin, subjectCos, clipSin, clipCos;
    ExpandedSinCos(subject.theta, subjectSin, subjectCos);
    ExpandedSinCos(clip.theta, clipSin, clipCos);
    const FloatExpansion sine = SubtractExpansions(MultiplyExpansions(subjectSin, clipCos),
                                                   MultiplyExpansions(subjectCos, clipSin));
    const FloatExpansion cosine = AddExpansions(MultiplyExpansions(subjectCos, clipCos),
                                                MultiplyExpansions(subjectSin, clipSin));
    FloatExpansion dx, dy, width, height, clipWidth, clipHeight;
    if constexpr (Trans) {
        dx = SubtractExpansions(EndpointCenter(subject.lowerX, subject.upperX),
                                EndpointCenter(clip.lowerX, clip.upperX));
        dy = SubtractExpansions(EndpointCenter(subject.lowerY, subject.upperY),
                                EndpointCenter(clip.lowerY, clip.upperY));
        width = EndpointExtent(subject.lowerX, subject.upperX);
        height = EndpointExtent(subject.lowerY, subject.upperY);
        clipWidth = EndpointExtent(clip.lowerX, clip.upperX);
        clipHeight = EndpointExtent(clip.lowerY, clip.upperY);
    } else {
        dx = SubtractExpansions({subject.centerX, 0.0F}, {clip.centerX, 0.0F});
        dy = SubtractExpansions({subject.centerY, 0.0F}, {clip.centerY, 0.0F});
        width = {subject.width, 0.0F};
        height = {subject.height, 0.0F};
        clipWidth = {clip.width, 0.0F};
        clipHeight = {clip.height, 0.0F};
    }
    const FloatExpansion centerX = AddExpansions(MultiplyExpansions(dx, clipCos), MultiplyExpansions(dy, clipSin));
    const FloatExpansion centerY = SubtractExpansions(MultiplyExpansions(dy, clipCos), MultiplyExpansions(dx, clipSin));
    constexpr uint32_t capacity = kIntersectionCapacity;
    float polygon[capacity * 4U];
    float output[capacity * 4U];
    float minX = kMaxFinite;
    float minY = kMaxFinite;
    float maxX = -kMaxFinite;
    float maxY = -kMaxFinite;
    for (uint32_t i = 0; i < 4; ++i) {
        const FloatExpansion x = MultiplyExpansion(width, (i == 0 || i == 3) ? -0.5F : 0.5F);
        const FloatExpansion y = MultiplyExpansion(height, i < 2 ? -0.5F : 0.5F);
        ExpandedPoint point;
        point.x = AddExpansions(centerX,
                                SubtractExpansions(MultiplyExpansions(x, cosine), MultiplyExpansions(y, sine)));
        point.y = AddExpansions(centerY, AddExpansions(MultiplyExpansions(x, sine), MultiplyExpansions(y, cosine)));
        StoreExpandedPoint(polygon, i, point);
        minX = fminf(minX, EvaluateExpansion(point.x));
        maxX = fmaxf(maxX, EvaluateExpansion(point.x));
        minY = fminf(minY, EvaluateExpansion(point.y));
        maxY = fmaxf(maxY, EvaluateExpansion(point.y));
    }
    const float halfClipWidth = 0.5F * EvaluateExpansion(clipWidth);
    const float halfClipHeight = 0.5F * EvaluateExpansion(clipHeight);
    const float boundScale = fabsf(EvaluateExpansion(centerX)) + fabsf(EvaluateExpansion(centerY)) +
                             EvaluateExpansion(width) + EvaluateExpansion(height) + halfClipWidth + halfClipHeight;
    const float boundPadding = boundScale * kBoundPaddingFactor;
    // Only classify well-separated bounds. Near-contact and non-finite bounds
    // retain the compensated clipping path; no small intersection is zeroed.
    if (isfinite(boundScale) && boundPadding >= kMinNormal) {
        if (minX > halfClipWidth + boundPadding || maxX < -halfClipWidth - boundPadding ||
            minY > halfClipHeight + boundPadding || maxY < -halfClipHeight - boundPadding) {
            return 0.0F;
        }
        if (minX > -halfClipWidth + boundPadding && maxX < halfClipWidth - boundPadding &&
            minY > -halfClipHeight + boundPadding && maxY < halfClipHeight - boundPadding) {
            return EvaluateExpansion(MultiplyExpansions(width, height));
        }
    }
    uint32_t count = 4;
    for (uint32_t edge = 0; edge < 4 && count != 0; ++edge) {
        const bool xAxis = edge < 2;
        const bool lower = (edge % 2) == 0;
        const FloatExpansion bound = MultiplyExpansion(xAxis ? clipWidth : clipHeight, lower ? -0.5F : 0.5F);
        ExpandedPoint previous = LoadExpandedPoint(polygon, count - 1);
        FloatExpansion previousDistance = SubtractExpansions(xAxis ? previous.x : previous.y, bound);
        if (!lower) {
            previousDistance = NegateExpansion(previousDistance);
        }
        bool previousInside = ExpansionNonnegative(previousDistance);
        uint32_t outputCount = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const ExpandedPoint current = LoadExpandedPoint(polygon, i);
            FloatExpansion distance = SubtractExpansions(xAxis ? current.x : current.y, bound);
            if (!lower) {
                distance = NegateExpansion(distance);
            }
            const bool inside = ExpansionNonnegative(distance);
            if (inside != previousInside && outputCount < capacity) {
                const FloatExpansion ratio = DivideExpansions(previousDistance,
                                                              SubtractExpansions(previousDistance, distance));
                ExpandedPoint intersection;
                intersection.x = xAxis ?
                                     bound :
                                     AddExpansions(previous.x, MultiplyExpansions(
                                                                   ratio, SubtractExpansions(current.x, previous.x)));
                intersection.y = xAxis ?
                                     AddExpansions(previous.y, MultiplyExpansions(
                                                                   ratio, SubtractExpansions(current.y, previous.y))) :
                                     bound;
                StoreExpandedPoint(output, outputCount++, intersection);
            }
            if (inside && outputCount < capacity) {
                StoreExpandedPoint(output, outputCount++, current);
            }
            previous = current;
            previousDistance = distance;
            previousInside = inside;
        }
        count = outputCount;
        for (uint32_t i = 0; i < count; ++i) {
            StoreExpandedPoint(polygon, i, LoadExpandedPoint(output, i));
        }
    }
    if (count < 3) {
        return 0.0F;
    }
    FloatExpansion area = {0.0F, 0.0F};
    for (uint32_t i = 1; i + 1 < count; ++i) {
        const ExpandedPoint origin = LoadExpandedPoint(polygon, 0);
        const ExpandedPoint first = LoadExpandedPoint(polygon, i);
        const ExpandedPoint second = LoadExpandedPoint(polygon, i + 1);
        const FloatExpansion ax = SubtractExpansions(first.x, origin.x);
        const FloatExpansion ay = SubtractExpansions(first.y, origin.y);
        const FloatExpansion bx = SubtractExpansions(second.x, origin.x);
        const FloatExpansion by = SubtractExpansions(second.y, origin.y);
        area = AddExpansions(area, SubtractExpansions(MultiplyExpansions(ax, by), MultiplyExpansions(ay, bx)));
    }
    return fabsf(EvaluateExpansion(MultiplyExpansion(area, 0.5F)));
}

template <bool Trans, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(kRotatedOverlapsSimtThreadNum) inline void RotatedOverlapsPairSimt(
    IndexT totalPairs, IndexT numBoxes, IndexT numQueries, __gm__ float* boxes, __gm__ float* queryBoxes,
    __gm__ float* overlaps)
{
    const IndexT pairsPerBatch = numBoxes * numQueries;
    for (IndexT pairIndex = static_cast<IndexT>(blockIdx.x * blockDim.x + threadIdx.x); pairIndex < totalPairs;
         pairIndex += static_cast<IndexT>(blockDim.x * gridDim.x)) {
        const IndexT batch = pairIndex / pairsPerBatch;
        const IndexT indexInBatch = pairIndex - batch * pairsPerBatch;
        const IndexT boxIndex = indexInBatch / numQueries;
        const IndexT queryIndex = indexInBatch - boxIndex * numQueries;
        PairBox box;
        PairBox query;
        LoadPairBox<Trans>(boxes, static_cast<uint64_t>(numBoxes), static_cast<uint64_t>(batch),
                           static_cast<uint64_t>(boxIndex), box);
        LoadPairBox<Trans>(queryBoxes, static_cast<uint64_t>(numQueries), static_cast<uint64_t>(batch),
                           static_cast<uint64_t>(queryIndex), query);
        if (!box.valid || !query.valid) {
            overlaps[pairIndex] = 0.0F;
            continue;
        }

        if (box.theta == 0.0F && query.theta == 0.0F) {
            overlaps[pairIndex] = AxisAlignedIntersectionArea<Trans>(box, query);
            continue;
        }

        if (DefinitelyDisjoint(box, query)) {
            overlaps[pairIndex] = 0.0F;
            continue;
        }

        // Clipping the smaller rectangle against the larger one reduces the
        // coordinate/cancellation error of narrow intersection polygons and
        // usually reduces the number of intermediate vertices as well.
        const float boxArea = box.width * box.height;
        const float queryArea = query.width * query.height;
        const bool boxIsSmaller = boxArea <= queryArea;
        overlaps[pairIndex] = PreciseIntersectionArea<Trans>(boxIsSmaller ? box : query, boxIsSmaller ? query : box);
    }
}

template <bool Trans, bool Use32Bit>
__aicore__ inline void ProcessPairParallelSimt(GM_ADDR boxes, GM_ADDR queryBoxes, GM_ADDR overlaps,
                                               const RotatedOverlapsTilingData* tilingData)
{
    using IndexT = typename IndexTypeSelector<Use32Bit>::type;
    asc_vf_call<RotatedOverlapsPairSimt<Trans, IndexT>>(
        dim3(kRotatedOverlapsSimtThreadNum), static_cast<IndexT>(tilingData->totalPairs),
        static_cast<IndexT>(tilingData->numBoxes), static_cast<IndexT>(tilingData->numQueries), (__gm__ float*)boxes,
        (__gm__ float*)queryBoxes, (__gm__ float*)overlaps);
}

template <bool Trans, bool Use32Bit>
class RotatedOverlapsKernel {
public:
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR queryBoxes, GM_ADDR overlaps,
                                const RotatedOverlapsTilingData* tilingData)
    {
        tilingData_ = tilingData;
        numBoxes_ = tilingData_->numBoxes;
        numQueries_ = tilingData_->numQueries;
        totalTasks_ = tilingData_->totalTasks;
        tasksPerCore_ = tilingData_->tasksPerCore;
        tileLen_ = tilingData_->tileLen;
        tilesPerOuter_ = tilingData_->tilesPerOuter;
        vectorizeBoxes_ = tilingData_->vectorizeBoxes != 0U;
        alignedTileLen_ = AlignElements(tileLen_);

        boxesGm_.SetGlobalBuffer((__gm__ float*)boxes);
        queryBoxesGm_.SetGlobalBuffer((__gm__ float*)queryBoxes);
        overlapsGm_.SetGlobalBuffer((__gm__ float*)overlaps);

        pipe_.InitBuffer(vectorBuffer_,
                         static_cast<uint64_t>(kRotatedOverlapsFloatVectorCount) * alignedTileLen_ * sizeof(float));
        pipe_.InitBuffer(mathBuffer_, tilingData_->mathTmpBytes);
        pipe_.InitBuffer(maskBuffer_, kRotatedOverlapsMaskReserveBytes);
        vectors_ = vectorBuffer_.Get<float>();
        mathTmp_ = mathBuffer_.Get<uint8_t>();
        masks_ = maskBuffer_.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        using IndexT = typename IndexTypeSelector<Use32Bit>::type;
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t taskStart64 = blockIndex * tasksPerCore_;
        if (taskStart64 >= totalTasks_) {
            return;
        }
        uint64_t taskEnd64 = taskStart64 + tasksPerCore_;
        if (taskEnd64 > totalTasks_) {
            taskEnd64 = totalTasks_;
        }

        // The host selects the 32-bit template only when task and output
        // offsets fit. Input channel bases stay uint64_t because their
        // five-channel strides may be wider than the logical output.
        const IndexT taskStart = static_cast<IndexT>(taskStart64);
        const IndexT taskEnd = static_cast<IndexT>(taskEnd64);
        for (IndexT task = taskStart; task < taskEnd; ++task) {
            if (vectorizeBoxes_) {
                ProcessBoxVectorTask(static_cast<uint64_t>(task));
            } else {
                ProcessQueryVectorTask(static_cast<uint64_t>(task));
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignElements(uint32_t count) const
    {
        return (count + kAlignElements - 1U) / kAlignElements * kAlignElements;
    }

    __aicore__ inline LocalTensor<float> Vec(uint32_t slot) const { return vectors_[slot * alignedTileLen_]; }

    __aicore__ inline LocalTensor<uint8_t> Mask(uint32_t slot) const { return masks_[slot * maskStride_]; }

    __aicore__ inline LocalTensor<float> ACornerX(uint32_t corner) const { return Vec(kACornerBase + corner * 2U); }

    __aicore__ inline LocalTensor<float> ACornerY(uint32_t corner) const
    {
        return Vec(kACornerBase + corner * 2U + 1U);
    }

    __aicore__ inline LocalTensor<float> BCornerX(uint32_t corner) const { return Vec(kBCornerBase + corner * 2U); }

    __aicore__ inline LocalTensor<float> BCornerY(uint32_t corner) const
    {
        return Vec(kBCornerBase + corner * 2U + 1U);
    }

    __aicore__ inline LocalTensor<float> CandidateX(uint32_t candidate) const
    {
        return Vec(kCandidateXBase + candidate);
    }

    __aicore__ inline LocalTensor<float> CandidateY(uint32_t candidate) const
    {
        return Vec(kCandidateYBase + candidate);
    }

    __aicore__ inline LocalTensor<float> CandidateKey(uint32_t candidate) const
    {
        return Vec(kCandidateKeyBase + candidate);
    }

    __aicore__ inline void AndMask(const LocalTensor<uint8_t>& dst, const LocalTensor<uint8_t>& src,
                                   uint32_t maskWordCount)
    {
        // Compare emits one bit per float lane.  Bitwise operations therefore
        // work on the compact mask storage, not on float-lane count.
        auto dst16 = dst.ReinterpretCast<uint16_t>();
        auto src16 = src.ReinterpretCast<uint16_t>();
        PipeBarrier<PIPE_V>();
        And(dst16, dst16, src16, static_cast<int32_t>(maskWordCount));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadAndNormaliseBroadcast(const GlobalTensor<float>& source, uint64_t sourceLength,
                                                     uint64_t batch, uint64_t sourceIndex, int32_t count)
    {
        const uint64_t batchBase = batch * kCoordinateCount * sourceLength;
        Duplicate(Vec(kAx), source.GetValue(batchBase + sourceIndex), count);
        Duplicate(Vec(kAy), source.GetValue(batchBase + sourceLength + sourceIndex), count);
        Duplicate(Vec(kAw), source.GetValue(batchBase + 2U * sourceLength + sourceIndex), count);
        Duplicate(Vec(kAh), source.GetValue(batchBase + 3U * sourceLength + sourceIndex), count);
        Duplicate(Vec(kAt), source.GetValue(batchBase + 4U * sourceLength + sourceIndex), count);
        Adds(Vec(kARawTheta), Vec(kAt), 0.0F, count);
        Normalise(Vec(kAx), Vec(kAy), Vec(kAw), Vec(kAh), count);
    }

    __aicore__ inline void LoadAndNormaliseVector(const GlobalTensor<float>& source, uint64_t sourceLength,
                                                  uint64_t batch, uint64_t sourceOffset, uint32_t currentCount,
                                                  uint32_t alignedCount)
    {
        const int32_t computeCount = static_cast<int32_t>(alignedCount);
        // Fill the entire vector tile before DMA so Sin/Cos has safe,
        // 32-byte-aligned padding lanes on non-aligned K tails.
        Duplicate(Vec(kQx), 0.0F, computeCount);
        Duplicate(Vec(kQy), 0.0F, computeCount);
        Duplicate(Vec(kQw), 0.0F, computeCount);
        Duplicate(Vec(kQh), 0.0F, computeCount);
        Duplicate(Vec(kQt), 0.0F, computeCount);
        PipeBarrier<PIPE_ALL>();

        const uint64_t batchBase = batch * kCoordinateCount * sourceLength + sourceOffset;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(currentCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0F};
        DataCopyPad(Vec(kQx), source[batchBase], copyParams, padParams);
        DataCopyPad(Vec(kQy), source[batchBase + sourceLength], copyParams, padParams);
        DataCopyPad(Vec(kQw), source[batchBase + 2U * sourceLength], copyParams, padParams);
        DataCopyPad(Vec(kQh), source[batchBase + 3U * sourceLength], copyParams, padParams);
        DataCopyPad(Vec(kQt), source[batchBase + 4U * sourceLength], copyParams, padParams);
        const event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        Normalise(Vec(kQx), Vec(kQy), Vec(kQw), Vec(kQh), computeCount);
    }

    __aicore__ inline void ProcessQueryVectorTask(uint64_t task)
    {
        const uint64_t row = task / tilesPerOuter_;
        const uint64_t tileIndex = task - row * tilesPerOuter_;
        const uint64_t batch = row / numBoxes_;
        const uint64_t boxIndex = row - batch * numBoxes_;
        const uint64_t queryOffset = tileIndex * tileLen_;
        const uint64_t remaining = numQueries_ - queryOffset;
        const uint32_t currentCount = static_cast<uint32_t>(remaining < tileLen_ ? remaining : tileLen_);
        const uint32_t alignedCount = AlignElements(currentCount);
        LoadAndNormaliseBroadcast(boxesGm_, numBoxes_, batch, boxIndex, static_cast<int32_t>(alignedCount));
        LoadAndNormaliseVector(queryBoxesGm_, numQueries_, batch, queryOffset, currentCount, alignedCount);
        ComputeTile(alignedCount);
        CopyOutput(row * numQueries_ + queryOffset, currentCount);
    }

    __aicore__ inline void ProcessBoxVectorTask(uint64_t task)
    {
        const uint64_t queryRow = task / tilesPerOuter_;
        const uint64_t tileIndex = task - queryRow * tilesPerOuter_;
        const uint64_t batch = queryRow / numQueries_;
        const uint64_t queryIndex = queryRow - batch * numQueries_;
        const uint64_t boxOffset = tileIndex * tileLen_;
        const uint64_t remaining = numBoxes_ - boxOffset;
        const uint32_t currentCount = static_cast<uint32_t>(remaining < tileLen_ ? remaining : tileLen_);
        const uint32_t alignedCount = AlignElements(currentCount);
        LoadAndNormaliseBroadcast(queryBoxesGm_, numQueries_, batch, queryIndex, static_cast<int32_t>(alignedCount));
        LoadAndNormaliseVector(boxesGm_, numBoxes_, batch, boxOffset, currentCount, alignedCount);
        ComputeTile(alignedCount);
        const uint64_t outputOffset = (batch * numBoxes_ + boxOffset) * numQueries_ + queryIndex;
        CopyOutputStrided(outputOffset, currentCount, alignedCount);
    }

    __aicore__ inline void Normalise(const LocalTensor<float>& x, const LocalTensor<float>& y,
                                     const LocalTensor<float>& w, const LocalTensor<float>& h, int32_t count)
    {
        if constexpr (Trans) {
            // [x1, y1, x2, y2] -> [cx, cy, w, h], all as vector operations.
            Sub(Vec(kTmp0), w, x, count);
            Sub(Vec(kTmp1), h, y, count);
            Add(Vec(kTmp2), x, w, count);
            Muls(x, Vec(kTmp2), kHalf, count);
            Add(Vec(kTmp2), y, h, count);
            Muls(y, Vec(kTmp2), kHalf, count);
            Adds(w, Vec(kTmp0), 0.0F, count);
            Adds(h, Vec(kTmp1), 0.0F, count);
        }
    }

    __aicore__ inline void ApplyFiniteCheck(const LocalTensor<float>& valid, const LocalTensor<float>& value,
                                            const LocalTensor<float>& zero, int32_t count)
    {
        Abs(Vec(kTmp0), value, count);
        CompareScalar(Mask(2U), Vec(kTmp0), kMaxFinite, CMPMODE::LE, static_cast<uint32_t>(count));
        Select(Vec(kTmp0), Mask(2U), Vec(kOne), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Mul(valid, valid, Vec(kTmp0), count);
    }

    __aicore__ inline void BuildValidity(const LocalTensor<float>& valid, const LocalTensor<float>& x,
                                         const LocalTensor<float>& y, const LocalTensor<float>& w,
                                         const LocalTensor<float>& h, const LocalTensor<float>& theta,
                                         const LocalTensor<float>& zero, int32_t count)
    {
        Duplicate(valid, 1.0F, count);
        ApplyFiniteCheck(valid, x, zero, count);
        ApplyFiniteCheck(valid, y, zero, count);
        ApplyFiniteCheck(valid, w, zero, count);
        ApplyFiniteCheck(valid, h, zero, count);
        ApplyFiniteCheck(valid, theta, zero, count);
        CompareScalar(Mask(2U), w, 0.0F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(Vec(kTmp0), Mask(2U), Vec(kOne), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Mul(valid, valid, Vec(kTmp0), count);
        CompareScalar(Mask(2U), h, 0.0F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(Vec(kTmp0), Mask(2U), Vec(kOne), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Mul(valid, valid, Vec(kTmp0), count);
    }

    __aicore__ inline void SanitiseBox(const LocalTensor<float>& valid, const LocalTensor<float>& x,
                                       const LocalTensor<float>& y, const LocalTensor<float>& w,
                                       const LocalTensor<float>& h, const LocalTensor<float>& theta,
                                       const LocalTensor<float>& zero, int32_t count)
    {
        CompareScalar(Mask(2U), valid, 0.5F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(x, Mask(2U), x, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Select(y, Mask(2U), y, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Select(w, Mask(2U), w, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Select(h, Mask(2U), h, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Select(theta, Mask(2U), theta, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
    }

    __aicore__ inline void BuildCorners(const LocalTensor<float>& cx, const LocalTensor<float>& cy,
                                        const LocalTensor<float>& width, const LocalTensor<float>& height,
                                        const LocalTensor<float>& sine, const LocalTensor<float>& cosine,
                                        uint32_t cornerBase, int32_t count)
    {
        Muls(Vec(kTmp0), width, kHalf, count);
        Muls(Vec(kTmp1), height, kHalf, count);
        constexpr float kSigns[kCornerCount][2] = {{-1.0F, -1.0F}, {1.0F, -1.0F}, {1.0F, 1.0F}, {-1.0F, 1.0F}};
        for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
            Muls(Vec(kTmp2), Vec(kTmp0), kSigns[corner][0], count);
            Muls(Vec(kTmp3), Vec(kTmp1), kSigns[corner][1], count);
            Mul(Vec(kTmp4), Vec(kTmp2), cosine, count);
            Mul(Vec(kTmp5), Vec(kTmp3), sine, count);
            Sub(Vec(kTmp6), Vec(kTmp4), Vec(kTmp5), count);
            Add(Vec(cornerBase + corner * 2U), cx, Vec(kTmp6), count);
            Mul(Vec(kTmp4), Vec(kTmp2), sine, count);
            Mul(Vec(kTmp5), Vec(kTmp3), cosine, count);
            Add(Vec(kTmp6), Vec(kTmp4), Vec(kTmp5), count);
            Add(Vec(cornerBase + corner * 2U + 1U), cy, Vec(kTmp6), count);
        }
    }

    __aicore__ inline void PointInConvexRect(const LocalTensor<float>& pointX, const LocalTensor<float>& pointY,
                                             uint32_t rectBase, const LocalTensor<float>& zero, int32_t count,
                                             uint32_t maskWordCount)
    {
        for (uint32_t edge = 0U; edge < kCornerCount; ++edge) {
            const uint32_t next = (edge + 1U) % kCornerCount;
            const LocalTensor<float> startX = Vec(rectBase + edge * 2U);
            const LocalTensor<float> startY = Vec(rectBase + edge * 2U + 1U);
            const LocalTensor<float> endX = Vec(rectBase + next * 2U);
            const LocalTensor<float> endY = Vec(rectBase + next * 2U + 1U);
            Sub(Vec(kTmp0), endX, startX, count);
            Sub(Vec(kTmp1), endY, startY, count);
            Sub(Vec(kTmp2), pointX, startX, count);
            Sub(Vec(kTmp3), pointY, startY, count);
            Mul(Vec(kTmp4), Vec(kTmp0), Vec(kTmp3), count);
            Mul(Vec(kTmp5), Vec(kTmp1), Vec(kTmp2), count);
            Sub(Vec(kTmp4), Vec(kTmp4), Vec(kTmp5), count);
            if (edge == 0U) {
                CompareScalar(Mask(1U), Vec(kTmp4), 0.0F, CMPMODE::GE, static_cast<uint32_t>(count));
            } else {
                CompareScalar(Mask(2U), Vec(kTmp4), 0.0F, CMPMODE::GE, static_cast<uint32_t>(count));
                AndMask(Mask(1U), Mask(2U), maskWordCount);
            }
        }
        (void)zero;
    }

    __aicore__ inline void AppendCandidate(uint32_t candidate, const LocalTensor<float>& sourceX,
                                           const LocalTensor<float>& sourceY, const LocalTensor<uint8_t>& validMask,
                                           const LocalTensor<float>& zero, int32_t count)
    {
        Select(CandidateX(candidate), validMask, sourceX, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateY(candidate), validMask, sourceY, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateKey(candidate), validMask, Vec(kOne), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Add(Vec(kCenterX), Vec(kCenterX), CandidateX(candidate), count);
        Add(Vec(kCenterY), Vec(kCenterY), CandidateY(candidate), count);
        Add(Vec(kAValid), Vec(kAValid), CandidateKey(candidate), count);
    }

    __aicore__ inline void AppendSegmentIntersection(
        uint32_t candidate, const LocalTensor<float>& ax0, const LocalTensor<float>& ay0, const LocalTensor<float>& ax1,
        const LocalTensor<float>& ay1, const LocalTensor<float>& bx0, const LocalTensor<float>& by0,
        const LocalTensor<float>& bx1, const LocalTensor<float>& by1, const LocalTensor<uint8_t>& pairMask,
        const LocalTensor<float>& zero, int32_t count, uint32_t maskWordCount)
    {
        // r=A1-A0, s=B1-B0, t=cross(B0-A0,s)/cross(r,s),
        // u=cross(B0-A0,r)/cross(r,s).  A safe denominator is selected
        // before vector division; its predicate remains in Mask(1).
        Sub(Vec(kTmp0), ax1, ax0, count); // rx
        Sub(Vec(kTmp1), ay1, ay0, count); // ry
        Sub(Vec(kTmp2), bx1, bx0, count); // sx
        Sub(Vec(kTmp3), by1, by0, count); // sy
        Mul(Vec(kTmp4), Vec(kTmp0), Vec(kTmp3), count);
        Mul(Vec(kTmp5), Vec(kTmp1), Vec(kTmp2), count);
        Sub(Vec(kTmp4), Vec(kTmp4), Vec(kTmp5), count); // denominator
        Abs(Vec(kTmp5), Vec(kTmp4), count);
        CompareScalar(Mask(1U), Vec(kTmp5), 0.0F, CMPMODE::GT, static_cast<uint32_t>(count));
        Sub(Vec(kTmp5), bx0, ax0, count); // qpx
        Sub(Vec(kTmp6), by0, ay0, count); // qpy
        Mul(Vec(kTmp7), Vec(kTmp5), Vec(kTmp3), count);
        Mul(CandidateX(candidate), Vec(kTmp6), Vec(kTmp2), count);
        Sub(Vec(kTmp7), Vec(kTmp7), CandidateX(candidate), count); // t numerator
        Mul(CandidateX(candidate), Vec(kTmp5), Vec(kTmp1), count);
        Mul(CandidateY(candidate), Vec(kTmp6), Vec(kTmp0), count);
        Sub(CandidateX(candidate), CandidateX(candidate), CandidateY(candidate), count); // u numerator
        Select(Vec(kTmp4), Mask(1U), Vec(kTmp4), Vec(kOne), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Div(Vec(kTmp7), Vec(kTmp7), Vec(kTmp4), count);                       // t
        Div(CandidateX(candidate), CandidateX(candidate), Vec(kTmp4), count); // u

        CompareScalar(Mask(2U), Vec(kTmp7), 0.0F, CMPMODE::GE, static_cast<uint32_t>(count));
        AndMask(Mask(1U), Mask(2U), maskWordCount);
        CompareScalar(Mask(2U), Vec(kTmp7), 1.0F, CMPMODE::LE, static_cast<uint32_t>(count));
        AndMask(Mask(1U), Mask(2U), maskWordCount);
        CompareScalar(Mask(2U), CandidateX(candidate), 0.0F, CMPMODE::GE, static_cast<uint32_t>(count));
        AndMask(Mask(1U), Mask(2U), maskWordCount);
        CompareScalar(Mask(2U), CandidateX(candidate), 1.0F, CMPMODE::LE, static_cast<uint32_t>(count));
        AndMask(Mask(1U), Mask(2U), maskWordCount);
        AndMask(Mask(1U), pairMask, maskWordCount);

        Mul(CandidateX(candidate), Vec(kTmp0), Vec(kTmp7), count);
        Add(CandidateX(candidate), CandidateX(candidate), ax0, count);
        Mul(CandidateY(candidate), Vec(kTmp1), Vec(kTmp7), count);
        Add(CandidateY(candidate), CandidateY(candidate), ay0, count);
        AppendCandidate(candidate, CandidateX(candidate), CandidateY(candidate), Mask(1U), zero, count);
    }

    __aicore__ inline void BuildCandidates(const LocalTensor<float>& zero, int32_t count)
    {
        // Compare/Select mask tensors themselves must start on a 32 B
        // boundary on DAV_3510.  A tile with only eight lanes still needs a
        // full aligned mask slice; only its first bit byte is semantically
        // consumed by Select(count).
        const uint32_t maskWordCount = kMaskStrideBytes / sizeof(uint16_t);
        Duplicate(Vec(kCenterX), 0.0F, count);
        Duplicate(Vec(kCenterY), 0.0F, count);
        // kAValid is no longer needed after pair validity has been built;
        // reuse it as the per-lane candidate count accumulator.
        Duplicate(Vec(kAValid), 0.0F, count);
        CompareScalar(Mask(0U), Vec(kPairValid), 0.5F, CMPMODE::GT, static_cast<uint32_t>(count));

        for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
            PointInConvexRect(ACornerX(corner), ACornerY(corner), kBCornerBase, zero, count, maskWordCount);
            AndMask(Mask(1U), Mask(0U), maskWordCount);
            AppendCandidate(corner, ACornerX(corner), ACornerY(corner), Mask(1U), zero, count);
        }
        for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
            PointInConvexRect(BCornerX(corner), BCornerY(corner), kACornerBase, zero, count, maskWordCount);
            AndMask(Mask(1U), Mask(0U), maskWordCount);
            AppendCandidate(kCornerCount + corner, BCornerX(corner), BCornerY(corner), Mask(1U), zero, count);
        }
        for (uint32_t aEdge = 0U; aEdge < kCornerCount; ++aEdge) {
            const uint32_t aNext = (aEdge + 1U) % kCornerCount;
            for (uint32_t bEdge = 0U; bEdge < kCornerCount; ++bEdge) {
                const uint32_t bNext = (bEdge + 1U) % kCornerCount;
                const uint32_t candidate = 8U + aEdge * kCornerCount + bEdge;
                AppendSegmentIntersection(candidate, ACornerX(aEdge), ACornerY(aEdge), ACornerX(aNext), ACornerY(aNext),
                                          BCornerX(bEdge), BCornerY(bEdge), BCornerX(bNext), BCornerY(bNext), Mask(0U),
                                          zero, count, maskWordCount);
            }
        }
        for (uint32_t candidate = kRealCandidateCount; candidate < kCandidateCount; ++candidate) {
            Duplicate(CandidateX(candidate), 0.0F, count);
            Duplicate(CandidateY(candidate), 0.0F, count);
            Duplicate(CandidateKey(candidate), 0.0F, count);
        }
    }

    __aicore__ inline void BuildPseudoAngleKeys(int32_t count)
    {
        Maxs(Vec(kTmp0), Vec(kAValid), 1.0F, count);
        Div(Vec(kCenterX), Vec(kCenterX), Vec(kTmp0), count);
        Div(Vec(kCenterY), Vec(kCenterY), Vec(kTmp0), count);
        for (uint32_t candidate = 0U; candidate < kCandidateCount; ++candidate) {
            // CandidateKey initially carries {0,1}; save its validity mask
            // before replacing it with the sortable pseudo-angle key.
            CompareScalar(Mask(1U), CandidateKey(candidate), 0.5F, CMPMODE::GT, static_cast<uint32_t>(count));
            Sub(Vec(kTmp0), CandidateX(candidate), Vec(kCenterX), count);
            Sub(Vec(kTmp1), CandidateY(candidate), Vec(kCenterY), count);
            Abs(Vec(kTmp2), Vec(kTmp0), count);
            Abs(Vec(kTmp3), Vec(kTmp1), count);
            Add(Vec(kTmp4), Vec(kTmp2), Vec(kTmp3), count);
            CompareScalar(Mask(2U), Vec(kTmp4), 0.0F, CMPMODE::GT, static_cast<uint32_t>(count));
            Select(Vec(kTmp4), Mask(2U), Vec(kTmp4), Vec(kOne), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<uint32_t>(count));
            Div(Vec(kTmp5), Vec(kTmp0), Vec(kTmp4), count);
            // Lower half: 1 + dx/(|dx|+|dy|); upper half: 3 - same.
            Adds(Vec(kTmp6), Vec(kTmp5), 1.0F, count);
            Muls(Vec(kTmp7), Vec(kTmp5), -1.0F, count);
            Adds(Vec(kTmp7), Vec(kTmp7), 3.0F, count);
            CompareScalar(Mask(2U), Vec(kTmp1), 0.0F, CMPMODE::LT, static_cast<uint32_t>(count));
            Select(Vec(kTmp6), Mask(2U), Vec(kTmp6), Vec(kTmp7), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<uint32_t>(count));
            Select(CandidateKey(candidate), Mask(1U), Vec(kTmp6), kInvalidKey, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   static_cast<uint32_t>(count));
        }
    }

    __aicore__ inline void CompareSwap(uint32_t left, uint32_t right, bool ascending, int32_t count)
    {
        // Keep original left values before any destination is updated.
        Adds(Vec(kSwapKey), CandidateKey(left), 0.0F, count);
        Adds(Vec(kSwapX), CandidateX(left), 0.0F, count);
        Adds(Vec(kSwapY), CandidateY(left), 0.0F, count);

        // The float32 pseudo-angle can quantise two distinct points to the
        // same key for an extremely thin polygon.  Resolve an equal-key pair
        // with the sign of cross(left-center, right-center), otherwise a
        // one-ulp-wide rectangle can be ordered as a self-crossing polygon
        // and lose half of its area.
        Compare(Mask(1U), CandidateKey(left), CandidateKey(right), ascending ? CMPMODE::GT : CMPMODE::LT,
                static_cast<uint32_t>(count));
        Select(Vec(kTmp6), Mask(1U), Vec(kOne), Vec(kOutput), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Sub(Vec(kTmp0), CandidateX(left), Vec(kCenterX), count);
        Sub(Vec(kTmp1), CandidateY(left), Vec(kCenterY), count);
        Sub(Vec(kTmp2), CandidateX(right), Vec(kCenterX), count);
        Sub(Vec(kTmp3), CandidateY(right), Vec(kCenterY), count);
        Mul(Vec(kTmp4), Vec(kTmp0), Vec(kTmp3), count);
        Mul(Vec(kTmp5), Vec(kTmp1), Vec(kTmp2), count);
        Sub(Vec(kTmp4), Vec(kTmp4), Vec(kTmp5), count);
        Compare(Mask(2U), CandidateKey(left), CandidateKey(right), CMPMODE::EQ, static_cast<uint32_t>(count));
        CompareScalar(Mask(0U), Vec(kTmp4), 0.0F, ascending ? CMPMODE::LT : CMPMODE::GT, static_cast<uint32_t>(count));
        AndMask(Mask(2U), Mask(0U), kMaskStrideBytes / sizeof(uint16_t));
        Select(Vec(kTmp7), Mask(2U), Vec(kOne), Vec(kOutput), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Add(Vec(kTmp6), Vec(kTmp6), Vec(kTmp7), count);
        CompareScalar(Mask(1U), Vec(kTmp6), 0.5F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(CandidateKey(left), Mask(1U), CandidateKey(right), CandidateKey(left), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateX(left), Mask(1U), CandidateX(right), CandidateX(left), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateY(left), Mask(1U), CandidateY(right), CandidateY(left), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateKey(right), Mask(1U), Vec(kSwapKey), CandidateKey(right), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateX(right), Mask(1U), Vec(kSwapX), CandidateX(right), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        Select(CandidateY(right), Mask(1U), Vec(kSwapY), CandidateY(right), SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
    }

    __aicore__ inline void BitonicSort(int32_t count)
    {
        // k/j/i bounds are compile-time constants.  They steer the fixed
        // vector network only; no branch depends on a pair's geometry.
        for (uint32_t k = 2U; k <= kCandidateCount; k <<= 1U) {
            for (uint32_t j = k >> 1U; j > 0U; j >>= 1U) {
                for (uint32_t i = 0U; i < kCandidateCount; ++i) {
                    const uint32_t partner = i ^ j;
                    if (partner > i) {
                        CompareSwap(i, partner, (i & k) == 0U, count);
                    }
                }
            }
        }
    }

    __aicore__ inline void PolygonArea(const LocalTensor<float>& zero, int32_t count)
    {
        // Valid keys are in [0,4], invalid keys are kInvalidKey.  Replacing
        // every invalid sorted slot with slot 0 turns the fixed 32-edge sum
        // into the exact closing edge of the valid polygon.
        for (uint32_t candidate = 0U; candidate < kCandidateCount; ++candidate) {
            CompareScalar(Mask(1U), CandidateKey(candidate), kInvalidKey, CMPMODE::LT, static_cast<uint32_t>(count));
            Select(CandidateX(candidate), Mask(1U), CandidateX(candidate), CandidateX(0U),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
            Select(CandidateY(candidate), Mask(1U), CandidateY(candidate), CandidateY(0U),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        }

        Duplicate(Vec(kOutput), 0.0F, count);
        for (uint32_t candidate = 0U; candidate < kCandidateCount; ++candidate) {
            const uint32_t next = (candidate + 1U) % kCandidateCount;
            // Shoelace area is translation invariant.  Working relative to
            // the candidate centroid avoids cancelling products around
            // |coordinate|^2 for otherwise small polygons at large absolute
            // coordinates, while keeping the entire calculation float32.
            Sub(Vec(kTmp0), CandidateX(candidate), Vec(kCenterX), count);
            Sub(Vec(kTmp1), CandidateY(candidate), Vec(kCenterY), count);
            Sub(Vec(kTmp2), CandidateX(next), Vec(kCenterX), count);
            Sub(Vec(kTmp3), CandidateY(next), Vec(kCenterY), count);
            Mul(Vec(kTmp4), Vec(kTmp0), Vec(kTmp3), count);
            Mul(Vec(kTmp5), Vec(kTmp1), Vec(kTmp2), count);
            Sub(Vec(kTmp4), Vec(kTmp4), Vec(kTmp5), count);
            Add(Vec(kOutput), Vec(kOutput), Vec(kTmp4), count);
        }
        Abs(Vec(kOutput), Vec(kOutput), count);
        Muls(Vec(kOutput), Vec(kOutput), kHalf, count);
        // Candidate count is a geometric validity condition. Do not apply an
        // area-scale epsilon here: every positive float32 intersection area is
        // part of the public result, including one-ulp-wide sliver overlaps.
        CompareScalar(Mask(2U), Vec(kAValid), 2.5F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(Vec(kOutput), Mask(2U), Vec(kOutput), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
        CompareScalar(Mask(1U), Vec(kPairValid), 0.5F, CMPMODE::GT, static_cast<uint32_t>(count));
        Select(Vec(kOutput), Mask(1U), Vec(kOutput), zero, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(count));
    }

    __aicore__ inline void ComputeTile(uint32_t alignedCount)
    {
        const int32_t count = static_cast<int32_t>(alignedCount);
        // The box vectors persist across query tiles.  ComputeTile consumes
        // kAt by converting it to radians, so restore its degree value before
        // validating and building the next tile's corners.
        Adds(Vec(kAt), Vec(kARawTheta), 0.0F, count);
        // output is the initial all-zero source for validity construction.
        Duplicate(Vec(kOutput), 0.0F, count);
        Duplicate(Vec(kOne), 1.0F, count);
        BuildValidity(Vec(kAValid), Vec(kAx), Vec(kAy), Vec(kAw), Vec(kAh), Vec(kAt), Vec(kOutput), count);
        BuildValidity(Vec(kQValid), Vec(kQx), Vec(kQy), Vec(kQw), Vec(kQh), Vec(kQt), Vec(kOutput), count);
        Mul(Vec(kPairValid), Vec(kAValid), Vec(kQValid), count);
        SanitiseBox(Vec(kAValid), Vec(kAx), Vec(kAy), Vec(kAw), Vec(kAh), Vec(kAt), Vec(kOutput), count);
        SanitiseBox(Vec(kQValid), Vec(kQx), Vec(kQy), Vec(kQw), Vec(kQh), Vec(kQt), Vec(kOutput), count);

        Muls(Vec(kAt), Vec(kAt), kDegreesToRadians, count);
        Muls(Vec(kQt), Vec(kQt), kDegreesToRadians, count);
        SinCos(Vec(kSinA), Vec(kCosA), Vec(kAt), mathTmp_, alignedCount);
        SinCos(Vec(kSinQ), Vec(kCosQ), Vec(kQt), mathTmp_, alignedCount);

        BuildCorners(Vec(kAx), Vec(kAy), Vec(kAw), Vec(kAh), Vec(kSinA), Vec(kCosA), kACornerBase, count);
        BuildCorners(Vec(kQx), Vec(kQy), Vec(kQw), Vec(kQh), Vec(kSinQ), Vec(kCosQ), kBCornerBase, count);

        // qValid is no longer required after pair validity has been formed;
        // retain a dedicated vector zero for candidate and final selections.
        Duplicate(Vec(kQValid), 0.0F, count);
        BuildCandidates(Vec(kQValid), count);
        BuildPseudoAngleKeys(count);
        BitonicSort(count);
        PolygonArea(Vec(kQValid), count);
    }

    __aicore__ inline void CopyOutput(uint64_t outputOffset, uint32_t currentCount)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(currentCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(overlapsGm_[outputOffset], Vec(kOutput), copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutputStrided(uint64_t outputOffset, uint32_t currentCount, uint32_t alignedCount)
    {
        PipeBarrier<PIPE_ALL>();
        // DataCopyPad advances a UB block by 32 bytes even when blockLen is a
        // single float. Scatter therefore expands the contiguous vector result
        // to one float per data block before the strided [N,K] copy-out.  Its
        // destination is an explicitly allocated eight-vector scratch region;
        // it cannot overlap the source, scatter offsets, or geometry slots.
        LocalTensor<int32_t> scatterOffset = Vec(kAh).template ReinterpretCast<int32_t>();
        CreateVecIndex(scatterOffset, 0, alignedCount);
        Muls(scatterOffset, scatterOffset, static_cast<int32_t>(32), alignedCount);
        Scatter(Vec(kScatterScratchBase), Vec(kOutput), scatterOffset.ReinterpretCast<uint32_t>(), 0U, alignedCount);
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams copyParams{static_cast<uint16_t>(currentCount), static_cast<uint32_t>(sizeof(float)), 0U,
                                     static_cast<uint32_t>((numQueries_ - 1U) * sizeof(float)), 0U};
        DataCopyPad(overlapsGm_[outputOffset], Vec(kScatterScratchBase), copyParams);
        const event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> vectorBuffer_;
    TBuf<TPosition::VECCALC> mathBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    LocalTensor<float> vectors_;
    LocalTensor<uint8_t> mathTmp_;
    LocalTensor<uint8_t> masks_;
    GlobalTensor<float> boxesGm_;
    GlobalTensor<float> queryBoxesGm_;
    GlobalTensor<float> overlapsGm_;
    const RotatedOverlapsTilingData* tilingData_{nullptr};
    uint64_t numBoxes_{0U};
    uint64_t numQueries_{0U};
    uint64_t totalTasks_{0U};
    uint64_t tasksPerCore_{0U};
    uint32_t tileLen_{0U};
    uint32_t tilesPerOuter_{0U};
    uint32_t alignedTileLen_{0U};
    uint32_t maskStride_{kMaskStrideBytes};
    bool vectorizeBoxes_{false};
};

} // namespace NsRotatedOverlaps

#endif // ROTATED_OVERLAPS_KERNEL_H_
