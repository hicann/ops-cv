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
 * A core owns complete [b, n, :] output rows.  A query tile is processed as
 * vector lanes: no output address has more than one writer and the geometric
 * path never reads a lane back through Scalar/GetValue.  For every lane we
 * create the fixed set of convex-intersection candidates:
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

__simt_callee__ inline float DegreesToRadians(float degrees)
{
    const float product = degrees * kDegreesToRadians;
    const float productError = fmaf(degrees, kDegreesToRadians, -product);
    return product + fmaf(degrees, kDegreesToRadiansLow, productError);
}

__simt_callee__ inline float AngleDifferenceToRadians(float first, float second)
{
    float differenceHigh = 0.0F;
    float differenceLow = 0.0F;
    TwoSum(first, -second, differenceHigh, differenceLow);
    const float product = differenceHigh * kDegreesToRadians;
    const float productError = fmaf(differenceHigh, kDegreesToRadians, -product);
    const float highCorrection = fmaf(differenceHigh, kDegreesToRadiansLow, productError);
    return product + fmaf(differenceLow, kDegreesToRadians, highCorrection);
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

__simt_callee__ inline float Cross(float firstX, float firstY, float secondX, float secondY)
{
    return firstX * secondY - firstY * secondX;
}

__simt_callee__ inline void BuildAxisAlignedCorners(const PairBox& box, float cornersX[kCornerCount],
                                                    float cornersY[kCornerCount])
{
    const float halfWidth = box.width * kHalf;
    const float halfHeight = box.height * kHalf;
    for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
        cornersX[corner] = (corner == 0U || corner == 3U) ? -halfWidth : halfWidth;
        cornersY[corner] = corner < 2U ? -halfHeight : halfHeight;
    }
}

__simt_callee__ inline void BuildRelativeCorners(const PairBox& box, float relativeCenterX, float relativeCenterY,
                                                 float relativeRadians, float cornersX[kCornerCount],
                                                 float cornersY[kCornerCount])
{
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(relativeRadians, &sine, &cosine);
    const float halfWidth = box.width * kHalf;
    const float halfHeight = box.height * kHalf;
    for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
        const float offsetX = (corner == 0U || corner == 3U) ? -halfWidth : halfWidth;
        const float offsetY = corner < 2U ? -halfHeight : halfHeight;
        cornersX[corner] = relativeCenterX + offsetX * cosine - offsetY * sine;
        cornersY[corner] = relativeCenterY + offsetX * sine + offsetY * cosine;
    }
}

__simt_callee__ inline bool BoxesAreDisjoint(const float firstX[kCornerCount], const float firstY[kCornerCount],
                                             const float secondX[kCornerCount], const float secondY[kCornerCount])
{
    float firstMinX = firstX[0];
    float firstMaxX = firstX[0];
    float firstMinY = firstY[0];
    float firstMaxY = firstY[0];
    float secondMinX = secondX[0];
    float secondMaxX = secondX[0];
    float secondMinY = secondY[0];
    float secondMaxY = secondY[0];
    for (uint32_t corner = 1U; corner < kCornerCount; ++corner) {
        firstMinX = fminf(firstMinX, firstX[corner]);
        firstMaxX = fmaxf(firstMaxX, firstX[corner]);
        firstMinY = fminf(firstMinY, firstY[corner]);
        firstMaxY = fmaxf(firstMaxY, firstY[corner]);
        secondMinX = fminf(secondMinX, secondX[corner]);
        secondMaxX = fmaxf(secondMaxX, secondX[corner]);
        secondMinY = fminf(secondMinY, secondY[corner]);
        secondMaxY = fmaxf(secondMaxY, secondY[corner]);
    }
    return firstMaxX < secondMinX || secondMaxX < firstMinX || firstMaxY < secondMinY || secondMaxY < firstMinY;
}

__simt_callee__ inline bool IsInsideClipEdge(float pointX, float pointY, float edgeStartX, float edgeStartY,
                                             float edgeEndX, float edgeEndY)
{
    return Cross(edgeEndX - edgeStartX, edgeEndY - edgeStartY, pointX - edgeStartX, pointY - edgeStartY) >= 0.0F;
}

__simt_callee__ inline void IntersectClipEdge(float startX, float startY, float endX, float endY, float edgeStartX,
                                              float edgeStartY, float edgeEndX, float edgeEndY, float& resultX,
                                              float& resultY)
{
    const float directionX = endX - startX;
    const float directionY = endY - startY;
    const float clipDirectionX = edgeEndX - edgeStartX;
    const float clipDirectionY = edgeEndY - edgeStartY;
    const float denominator = Cross(directionX, directionY, clipDirectionX, clipDirectionY);
    if (denominator == 0.0F) {
        resultX = endX;
        resultY = endY;
        return;
    }
    const float ratio = Cross(edgeStartX - startX, edgeStartY - startY, clipDirectionX, clipDirectionY) / denominator;
    resultX = startX + ratio * directionX;
    resultY = startY + ratio * directionY;
}

__simt_callee__ inline float ClipIntersectionArea(const float subjectX[kCornerCount],
                                                  const float subjectY[kCornerCount], const float clipX[kCornerCount],
                                                  const float clipY[kCornerCount])
{
    constexpr uint32_t kMaxIntersectionCorners = 8U;
    float polygonX[kMaxIntersectionCorners];
    float polygonY[kMaxIntersectionCorners];
    float clippedX[kMaxIntersectionCorners];
    float clippedY[kMaxIntersectionCorners];
    uint32_t polygonCount = kCornerCount;
    for (uint32_t corner = 0U; corner < kCornerCount; ++corner) {
        polygonX[corner] = subjectX[corner];
        polygonY[corner] = subjectY[corner];
    }

    for (uint32_t edge = 0U; edge < kCornerCount && polygonCount != 0U; ++edge) {
        const uint32_t edgeNext = (edge + 1U) % kCornerCount;
        const float edgeStartX = clipX[edge];
        const float edgeStartY = clipY[edge];
        const float edgeEndX = clipX[edgeNext];
        const float edgeEndY = clipY[edgeNext];
        uint32_t clippedCount = 0U;
        float previousX = polygonX[polygonCount - 1U];
        float previousY = polygonY[polygonCount - 1U];
        bool previousInside = IsInsideClipEdge(previousX, previousY, edgeStartX, edgeStartY, edgeEndX, edgeEndY);
        for (uint32_t current = 0U; current < polygonCount; ++current) {
            const float currentX = polygonX[current];
            const float currentY = polygonY[current];
            const bool currentInside = IsInsideClipEdge(currentX, currentY, edgeStartX, edgeStartY, edgeEndX, edgeEndY);
            if (currentInside != previousInside && clippedCount < kMaxIntersectionCorners) {
                IntersectClipEdge(previousX, previousY, currentX, currentY, edgeStartX, edgeStartY, edgeEndX, edgeEndY,
                                  clippedX[clippedCount], clippedY[clippedCount]);
                ++clippedCount;
            }
            if (currentInside && clippedCount < kMaxIntersectionCorners) {
                clippedX[clippedCount] = currentX;
                clippedY[clippedCount] = currentY;
                ++clippedCount;
            }
            previousX = currentX;
            previousY = currentY;
            previousInside = currentInside;
        }
        polygonCount = clippedCount;
        for (uint32_t corner = 0U; corner < polygonCount; ++corner) {
            polygonX[corner] = clippedX[corner];
            polygonY[corner] = clippedY[corner];
        }
    }

    if (polygonCount < 3U) {
        return 0.0F;
    }
    float doubledArea = 0.0F;
    const float originX = polygonX[0];
    const float originY = polygonY[0];
    for (uint32_t corner = 1U; corner + 1U < polygonCount; ++corner) {
        doubledArea += Cross(polygonX[corner] - originX, polygonY[corner] - originY, polygonX[corner + 1U] - originX,
                             polygonY[corner + 1U] - originY);
    }
    return fabsf(doubledArea) * kHalf;
}

__simt_callee__ inline float RotatedIntersectionArea(const PairBox& subject, const PairBox& clip)
{
    const float relativeClipX = clip.centerX - subject.centerX;
    const float relativeClipY = clip.centerY - subject.centerY;
    if (!isfinite(relativeClipX) || !isfinite(relativeClipY)) {
        return 0.0F;
    }

    float subjectCornersX[kCornerCount];
    float subjectCornersY[kCornerCount];
    float clipCornersX[kCornerCount];
    float clipCornersY[kCornerCount];
    float clipSine = 0.0F;
    float clipCosine = 0.0F;
    sincosf(DegreesToRadians(clip.theta), &clipSine, &clipCosine);
    const float relativeSubjectX = -relativeClipX;
    const float relativeSubjectY = -relativeClipY;
    const float alignedSubjectX = fmaf(relativeSubjectY, clipSine, relativeSubjectX * clipCosine);
    const float alignedSubjectY = fmaf(-relativeSubjectX, clipSine, relativeSubjectY * clipCosine);
    BuildRelativeCorners(subject, alignedSubjectX, alignedSubjectY, AngleDifferenceToRadians(subject.theta, clip.theta),
                         subjectCornersX, subjectCornersY);
    BuildAxisAlignedCorners(clip, clipCornersX, clipCornersY);
    return BoxesAreDisjoint(subjectCornersX, subjectCornersY, clipCornersX, clipCornersY) ?
               0.0F :
               ClipIntersectionArea(subjectCornersX, subjectCornersY, clipCornersX, clipCornersY);
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

        // Clipping the smaller rectangle against the larger one reduces the
        // coordinate/cancellation error of narrow intersection polygons and
        // usually reduces the number of intermediate vertices as well.
        const float boxArea = box.width * box.height;
        const float queryArea = query.width * query.height;
        overlaps[pairIndex] = boxArea <= queryArea ? RotatedIntersectionArea(box, query) :
                                                     RotatedIntersectionArea(query, box);
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
