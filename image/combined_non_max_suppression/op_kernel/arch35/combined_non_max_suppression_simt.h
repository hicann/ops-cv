/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMBINED_NON_MAX_SUPPRESSION_SIMT_H_
#define COMBINED_NON_MAX_SUPPRESSION_SIMT_H_

#include "kernel_operator.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#include "combined_non_max_suppression_tiling_data.h"

namespace CombinedNonMaxSuppressionOps {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t MIN_THREAD_NUM = 32;
constexpr uint32_t MAX_NUM_CLASSES = 200;
constexpr uint32_t HOT_UB_MAX_BOXES = 4096;
constexpr uint32_t UB_ALIGN_BYTES = 32;
constexpr float NEG_INF = -3.402823466e+38F;

__aicore__ inline uint32_t AlignUbBytes(uint32_t bytes)
{
    return (bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES * UB_ALIGN_BYTES;
}

__aicore__ inline uint32_t GetSimtThreadNum(int32_t workItems)
{
    uint32_t threadNum = MIN_THREAD_NUM;
    while (threadNum < static_cast<uint32_t>(workItems) && threadNum < THREAD_NUM) {
        threadNum <<= 1;
    }
    return threadNum;
}

__simt_callee__ __aicore__ inline float MinFloat(float a, float b) { return a < b ? a : b; }

__simt_callee__ __aicore__ inline float MaxFloat(float a, float b) { return a > b ? a : b; }

__simt_callee__ __aicore__ inline bool IsBetter(float lhsScore, int32_t lhsIndex, float rhsScore, int32_t rhsIndex)
{
    return lhsScore > rhsScore || (lhsScore == rhsScore && lhsIndex >= 0 && (rhsIndex < 0 || lhsIndex < rhsIndex));
}

__simt_callee__ __aicore__ inline float CalcIou(__gm__ const float* boxes, int64_t lhsOffset, int64_t rhsOffset)
{
    const float lhsYMin = MinFloat(boxes[lhsOffset], boxes[lhsOffset + 2]);
    const float lhsXMin = MinFloat(boxes[lhsOffset + 1], boxes[lhsOffset + 3]);
    const float lhsYMax = MaxFloat(boxes[lhsOffset], boxes[lhsOffset + 2]);
    const float lhsXMax = MaxFloat(boxes[lhsOffset + 1], boxes[lhsOffset + 3]);
    const float rhsYMin = MinFloat(boxes[rhsOffset], boxes[rhsOffset + 2]);
    const float rhsXMin = MinFloat(boxes[rhsOffset + 1], boxes[rhsOffset + 3]);
    const float rhsYMax = MaxFloat(boxes[rhsOffset], boxes[rhsOffset + 2]);
    const float rhsXMax = MaxFloat(boxes[rhsOffset + 1], boxes[rhsOffset + 3]);

    const float intersectH = MaxFloat(MinFloat(lhsYMax, rhsYMax) - MaxFloat(lhsYMin, rhsYMin), 0.0F);
    const float intersectW = MaxFloat(MinFloat(lhsXMax, rhsXMax) - MaxFloat(lhsXMin, rhsXMin), 0.0F);
    const float intersection = intersectH * intersectW;
    const float lhsArea = MaxFloat(lhsYMax - lhsYMin, 0.0F) * MaxFloat(lhsXMax - lhsXMin, 0.0F);
    const float rhsArea = MaxFloat(rhsYMax - rhsYMin, 0.0F) * MaxFloat(rhsXMax - rhsXMin, 0.0F);
    const float unionArea = lhsArea + rhsArea - intersection;
    return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

__simt_callee__ __aicore__ inline float CalcIouUb(__ubuf__ const float* boxes, int32_t lhsOffset, int32_t rhsOffset)
{
    const float lhsYMin = MinFloat(boxes[lhsOffset], boxes[lhsOffset + 2]);
    const float lhsXMin = MinFloat(boxes[lhsOffset + 1], boxes[lhsOffset + 3]);
    const float lhsYMax = MaxFloat(boxes[lhsOffset], boxes[lhsOffset + 2]);
    const float lhsXMax = MaxFloat(boxes[lhsOffset + 1], boxes[lhsOffset + 3]);
    const float rhsYMin = MinFloat(boxes[rhsOffset], boxes[rhsOffset + 2]);
    const float rhsXMin = MinFloat(boxes[rhsOffset + 1], boxes[rhsOffset + 3]);
    const float rhsYMax = MaxFloat(boxes[rhsOffset], boxes[rhsOffset + 2]);
    const float rhsXMax = MaxFloat(boxes[rhsOffset + 1], boxes[rhsOffset + 3]);

    const float intersectH = MaxFloat(MinFloat(lhsYMax, rhsYMax) - MaxFloat(lhsYMin, rhsYMin), 0.0F);
    const float intersectW = MaxFloat(MinFloat(lhsXMax, rhsXMax) - MaxFloat(lhsXMin, rhsXMin), 0.0F);
    const float intersection = intersectH * intersectW;
    const float lhsArea = MaxFloat(lhsYMax - lhsYMin, 0.0F) * MaxFloat(lhsXMax - lhsXMin, 0.0F);
    const float rhsArea = MaxFloat(rhsYMax - rhsYMin, 0.0F) * MaxFloat(rhsXMax - rhsXMin, 0.0F);
    const float unionArea = lhsArea + rhsArea - intersection;
    return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

// Like Sort's non-last-axis gather path, this VF is only responsible for
// staging strided GM data. The iterative NMS hot loop below operates on UB.
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__
    void LoadTaskHotData(__gm__ const float* boxes, __gm__ const float* scores, __ubuf__ float* boxesUb,
                         __ubuf__ float* scoresUb, int32_t batchIdx, int32_t classIdx, int32_t numBoxes,
                         int32_t boxClasses, int32_t numClasses)
{
    const int32_t boxClass = boxClasses == 1 ? 0 : classIdx;
    for (int32_t anchor = static_cast<int32_t>(threadIdx.x); anchor < numBoxes;
         anchor += static_cast<int32_t>(blockDim.x)) {
        const int64_t scoreOffset = (static_cast<int64_t>(batchIdx) * numBoxes + anchor) * numClasses + classIdx;
        scoresUb[anchor] = scores[scoreOffset];
        const int64_t boxOffset = ((static_cast<int64_t>(batchIdx) * numBoxes + anchor) * boxClasses + boxClass) * 4;
        const int32_t ubOffset = anchor * 4;
        boxesUb[ubOffset] = boxes[boxOffset];
        boxesUb[ubOffset + 1] = boxes[boxOffset + 1];
        boxesUb[ubOffset + 2] = boxes[boxOffset + 2];
        boxesUb[ubOffset + 3] = boxes[boxOffset + 3];
    }
}

__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__
    void SelectClassNmsUb(__ubuf__ const float* boxes, __ubuf__ const float* scores, __ubuf__ float* selectedScores,
                          __ubuf__ int32_t* selectedIndices, __ubuf__ int32_t* selectedCount,
                          __ubuf__ uint8_t* suppressed, __ubuf__ float* reduceScores, __ubuf__ int32_t* reduceIndices,
                          int32_t numBoxes, int32_t maxOutputPerClass, float iouThreshold, float scoreThreshold)
{
    const uint32_t tid = threadIdx.x;
    for (int32_t index = static_cast<int32_t>(tid); index < numBoxes; index += static_cast<int32_t>(blockDim.x)) {
        suppressed[index] = 0;
    }
    for (int32_t index = static_cast<int32_t>(tid); index < maxOutputPerClass;
         index += static_cast<int32_t>(blockDim.x)) {
        selectedScores[index] = NEG_INF;
        selectedIndices[index] = -1;
    }
    if (tid == 0) {
        selectedCount[0] = 0;
    }
    asc_syncthreads();

    for (int32_t outputIndex = 0; outputIndex < maxOutputPerClass; ++outputIndex) {
        float localBestScore = NEG_INF;
        int32_t localBestIndex = -1;
        for (int32_t anchor = static_cast<int32_t>(tid); anchor < numBoxes;
             anchor += static_cast<int32_t>(blockDim.x)) {
            if (suppressed[anchor] == 0 && scores[anchor] > scoreThreshold &&
                IsBetter(scores[anchor], anchor, localBestScore, localBestIndex)) {
                localBestScore = scores[anchor];
                localBestIndex = anchor;
            }
        }
        reduceScores[tid] = localBestScore;
        reduceIndices[tid] = localBestIndex;
        asc_syncthreads();

        for (uint32_t stride = static_cast<uint32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride && IsBetter(reduceScores[tid + stride], reduceIndices[tid + stride], reduceScores[tid],
                                         reduceIndices[tid])) {
                reduceScores[tid] = reduceScores[tid + stride];
                reduceIndices[tid] = reduceIndices[tid + stride];
            }
            asc_syncthreads();
        }

        const int32_t bestIndex = reduceIndices[0];
        if (bestIndex < 0) {
            break;
        }
        if (tid == 0) {
            selectedScores[outputIndex] = reduceScores[0];
            selectedIndices[outputIndex] = bestIndex;
            selectedCount[0] = outputIndex + 1;
        }
        asc_syncthreads();

        const int32_t bestBoxOffset = bestIndex * 4;
        for (int32_t anchor = static_cast<int32_t>(tid); anchor < numBoxes;
             anchor += static_cast<int32_t>(blockDim.x)) {
            if (suppressed[anchor] == 0 &&
                (anchor == bestIndex || CalcIouUb(boxes, bestBoxOffset, anchor * 4) > iouThreshold)) {
                suppressed[anchor] = 1;
            }
        }
        asc_syncthreads();
    }
}

__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__
    void SelectClassNms(__gm__ const float* boxes, __gm__ const float* scores, __gm__ float* selectedScores,
                        __gm__ int32_t* selectedIndices, __gm__ int32_t* selectedCounts, __gm__ uint8_t* suppressed,
                        __ubuf__ float* reduceScores, __ubuf__ int32_t* reduceIndices, int32_t batchIdx,
                        int32_t classIdx, int32_t taskIdx, int32_t numBoxes, int32_t boxClasses, int32_t numClasses,
                        int32_t maxOutputPerClass, float iouThreshold, float scoreThreshold)
{
    const uint32_t tid = threadIdx.x;
    const int32_t selectedBase = taskIdx * maxOutputPerClass;
    for (int32_t index = static_cast<int32_t>(tid); index < numBoxes; index += static_cast<int32_t>(blockDim.x)) {
        suppressed[index] = 0;
    }
    for (int32_t index = static_cast<int32_t>(tid); index < maxOutputPerClass;
         index += static_cast<int32_t>(blockDim.x)) {
        selectedScores[selectedBase + index] = NEG_INF;
        selectedIndices[selectedBase + index] = -1;
    }
    if (tid == 0) {
        selectedCounts[taskIdx] = 0;
    }
    asc_syncthreads();

    const int32_t boxClass = boxClasses == 1 ? 0 : classIdx;
    int32_t selectedCount = 0;
    for (int32_t outputIndex = 0; outputIndex < maxOutputPerClass; ++outputIndex) {
        float localBestScore = NEG_INF;
        int32_t localBestIndex = -1;
        for (int32_t anchor = static_cast<int32_t>(tid); anchor < numBoxes;
             anchor += static_cast<int32_t>(blockDim.x)) {
            if (suppressed[anchor] != 0) {
                continue;
            }
            const int64_t scoreOffset = (static_cast<int64_t>(batchIdx) * numBoxes + anchor) * numClasses + classIdx;
            const float score = scores[scoreOffset];
            if (score > scoreThreshold && IsBetter(score, anchor, localBestScore, localBestIndex)) {
                localBestScore = score;
                localBestIndex = anchor;
            }
        }
        reduceScores[tid] = localBestScore;
        reduceIndices[tid] = localBestIndex;
        asc_syncthreads();

        for (uint32_t stride = static_cast<uint32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride && IsBetter(reduceScores[tid + stride], reduceIndices[tid + stride], reduceScores[tid],
                                         reduceIndices[tid])) {
                reduceScores[tid] = reduceScores[tid + stride];
                reduceIndices[tid] = reduceIndices[tid + stride];
            }
            asc_syncthreads();
        }

        const int32_t bestIndex = reduceIndices[0];
        if (bestIndex < 0) {
            break;
        }
        if (tid == 0) {
            selectedScores[selectedBase + outputIndex] = reduceScores[0];
            selectedIndices[selectedBase + outputIndex] = bestIndex;
            selectedCount = outputIndex + 1;
            selectedCounts[taskIdx] = selectedCount;
        }
        asc_syncthreads();

        const int64_t bestBoxOffset = ((static_cast<int64_t>(batchIdx) * numBoxes + bestIndex) * boxClasses +
                                       boxClass) *
                                      4;
        for (int32_t anchor = static_cast<int32_t>(tid); anchor < numBoxes;
             anchor += static_cast<int32_t>(blockDim.x)) {
            if (suppressed[anchor] != 0) {
                continue;
            }
            const int64_t boxOffset = ((static_cast<int64_t>(batchIdx) * numBoxes + anchor) * boxClasses + boxClass) *
                                      4;
            if (anchor == bestIndex || CalcIou(boxes, bestBoxOffset, boxOffset) > iouThreshold) {
                suppressed[anchor] = 1;
            }
        }
        asc_syncthreads();
    }
}

__simt_callee__ __aicore__ inline float ClipCoordinate(float value) { return MinFloat(MaxFloat(value, 0.0F), 1.0F); }

__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__
    void MergeBatchResults(__gm__ const float* boxes, __gm__ const float* selectedScores,
                           __gm__ const int32_t* selectedIndices, __gm__ const int32_t* selectedCounts,
                           __ubuf__ float* nmsedBoxes, __ubuf__ float* nmsedScores, __ubuf__ float* nmsedClasses,
                           __ubuf__ int32_t* validDetections, __ubuf__ float* reduceScores,
                           __ubuf__ int32_t* reduceIndices, __ubuf__ int32_t* classCursors,
                           __ubuf__ int32_t* classCounts, __ubuf__ float* headScores, __ubuf__ int32_t* headIndices,
                           int32_t batchIdx, int32_t numBoxes, int32_t boxClasses, int32_t numClasses,
                           int32_t maxOutputPerClass, int32_t outputSize, int32_t clipBoxes)
{
    const uint32_t tid = threadIdx.x;

    const int32_t candidateBase = batchIdx * numClasses * maxOutputPerClass;
    const int32_t taskBase = batchIdx * numClasses;
    for (int32_t classIdx = static_cast<int32_t>(tid); classIdx < numClasses;
         classIdx += static_cast<int32_t>(blockDim.x)) {
        classCursors[classIdx] = 0;
        const int32_t count = selectedCounts[taskBase + classIdx];
        classCounts[classIdx] = count;
        if (count > 0) {
            const int32_t candidate = classIdx * maxOutputPerClass;
            headScores[classIdx] = selectedScores[candidateBase + candidate];
            headIndices[classIdx] = selectedIndices[candidateBase + candidate];
        } else {
            headScores[classIdx] = NEG_INF;
            headIndices[classIdx] = -1;
        }
    }
    for (int32_t outputIndex = static_cast<int32_t>(tid); outputIndex < outputSize;
         outputIndex += static_cast<int32_t>(blockDim.x)) {
        nmsedScores[outputIndex] = 0.0F;
        nmsedClasses[outputIndex] = 0.0F;
        const int32_t boxOutputOffset = outputIndex * 4;
        nmsedBoxes[boxOutputOffset] = 0.0F;
        nmsedBoxes[boxOutputOffset + 1] = 0.0F;
        nmsedBoxes[boxOutputOffset + 2] = 0.0F;
        nmsedBoxes[boxOutputOffset + 3] = 0.0F;
    }
    if (tid == 0) {
        validDetections[0] = 0;
    }
    asc_syncthreads();

    for (int32_t outputIndex = 0; outputIndex < outputSize; ++outputIndex) {
        float localBestScore = NEG_INF;
        int32_t localBestIndex = -1;
        for (int32_t classIdx = static_cast<int32_t>(tid); classIdx < numClasses;
             classIdx += static_cast<int32_t>(blockDim.x)) {
            const int32_t cursor = classCursors[classIdx];
            if (cursor >= classCounts[classIdx]) {
                continue;
            }
            const int32_t candidate = classIdx * maxOutputPerClass + cursor;
            const float score = headScores[classIdx];
            if (IsBetter(score, candidate, localBestScore, localBestIndex)) {
                localBestScore = score;
                localBestIndex = candidate;
            }
        }
        reduceScores[tid] = localBestScore;
        reduceIndices[tid] = localBestIndex;
        asc_syncthreads();

        for (uint32_t stride = static_cast<uint32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride && IsBetter(reduceScores[tid + stride], reduceIndices[tid + stride], reduceScores[tid],
                                         reduceIndices[tid])) {
                reduceScores[tid] = reduceScores[tid + stride];
                reduceIndices[tid] = reduceIndices[tid + stride];
            }
            asc_syncthreads();
        }

        const int32_t bestCandidate = reduceIndices[0];
        if (bestCandidate < 0 || reduceScores[0] == NEG_INF) {
            break;
        }
        if (tid == 0) {
            const int32_t classIdx = bestCandidate / maxOutputPerClass;
            const int32_t anchor = headIndices[classIdx];
            const int32_t boxClass = boxClasses == 1 ? 0 : classIdx;
            const int64_t boxInputOffset = ((static_cast<int64_t>(batchIdx) * numBoxes + anchor) * boxClasses +
                                            boxClass) *
                                           4;
            const int32_t boxOutputOffset = outputIndex * 4;
            for (int32_t coord = 0; coord < 4; ++coord) {
                float value = boxes[boxInputOffset + coord];
                if (clipBoxes != 0) {
                    value = ClipCoordinate(value);
                }
                nmsedBoxes[boxOutputOffset + coord] = value;
            }
            nmsedScores[outputIndex] = reduceScores[0];
            nmsedClasses[outputIndex] = static_cast<float>(classIdx);
            const int32_t nextCursor = classCursors[classIdx] + 1;
            classCursors[classIdx] = nextCursor;
            if (nextCursor < classCounts[classIdx]) {
                const int32_t nextCandidate = classIdx * maxOutputPerClass + nextCursor;
                headScores[classIdx] = selectedScores[candidateBase + nextCandidate];
                headIndices[classIdx] = selectedIndices[candidateBase + nextCandidate];
            } else {
                headScores[classIdx] = NEG_INF;
                headIndices[classIdx] = -1;
            }
            validDetections[0] = outputIndex + 1;
        }
        asc_syncthreads();
    }
}

class CombinedNonMaxSuppressionKernel {
public:
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR nmsedBoxes, GM_ADDR nmsedScores,
                                GM_ADDR nmsedClasses, GM_ADDR validDetections, GM_ADDR workspace,
                                const CombinedNonMaxSuppressionTilingData* tiling, TPipe* pipe)
    {
        boxes_ = reinterpret_cast<__gm__ float*>(boxes);
        scores_ = reinterpret_cast<__gm__ float*>(scores);
        nmsedBoxes_ = reinterpret_cast<__gm__ float*>(nmsedBoxes);
        nmsedScores_ = reinterpret_cast<__gm__ float*>(nmsedScores);
        nmsedClasses_ = reinterpret_cast<__gm__ float*>(nmsedClasses);
        validDetections_ = reinterpret_cast<__gm__ int32_t*>(validDetections);
        workspace_ = reinterpret_cast<__gm__ uint8_t*>(workspace);
        tiling_ = tiling;
        coreIdx_ = GetBlockIdx();
        pipe_ = pipe;
        useHotUb_ = tiling_->numBoxes <= static_cast<int32_t>(HOT_UB_MAX_BOXES);
        const uint32_t hotBoxCount = useHotUb_ ? static_cast<uint32_t>(tiling_->numBoxes) : 1U;
        pipe_->InitBuffer(scratchBuffer_, THREAD_NUM * (sizeof(float) + sizeof(int32_t)));
        pipe_->InitBuffer(hotBoxesBuffer_, AlignUbBytes(hotBoxCount * 4U * sizeof(float)));
        pipe_->InitBuffer(hotScoresBuffer_, AlignUbBytes(hotBoxCount * sizeof(float)));
        pipe_->InitBuffer(hotSuppressedBuffer_, AlignUbBytes(hotBoxCount * sizeof(uint8_t)));
        pipe_->InitBuffer(selectedScoresBuffer_,
                          AlignUbBytes(static_cast<uint32_t>(tiling_->maxOutputPerClass) * sizeof(float)));
        pipe_->InitBuffer(selectedIndicesBuffer_,
                          AlignUbBytes(static_cast<uint32_t>(tiling_->maxOutputPerClass) * sizeof(int32_t)));
        pipe_->InitBuffer(selectedCountBuffer_, UB_ALIGN_BYTES);
        pipe_->InitBuffer(mergeStateBuffer_, AlignUbBytes(MAX_NUM_CLASSES * 4U * sizeof(int32_t)));
        pipe_->InitBuffer(outputBoxesBuffer_,
                          AlignUbBytes(static_cast<uint32_t>(tiling_->outputSize) * 4U * sizeof(float)));
        pipe_->InitBuffer(outputScoresBuffer_,
                          AlignUbBytes(static_cast<uint32_t>(tiling_->outputSize) * sizeof(float)));
        pipe_->InitBuffer(outputClassesBuffer_,
                          AlignUbBytes(static_cast<uint32_t>(tiling_->outputSize) * sizeof(float)));
        pipe_->InitBuffer(validDetectionsBuffer_, UB_ALIGN_BYTES);
        eventVToMte3_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        eventMte3ToV_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint8_t> scratch = scratchBuffer_.Get<uint8_t>();
        __ubuf__ float* reduceScores = reinterpret_cast<__ubuf__ float*>(scratch.GetPhyAddr());
        __ubuf__ int32_t* reduceIndices = reinterpret_cast<__ubuf__ int32_t*>(
            scratch.GetPhyAddr(THREAD_NUM * sizeof(float)));
        __gm__ float* selectedScores = reinterpret_cast<__gm__ float*>(workspace_ + tiling_->selectedScoresOffset);
        __gm__ int32_t* selectedIndices = reinterpret_cast<__gm__ int32_t*>(workspace_ +
                                                                            tiling_->selectedIndicesOffset);
        __gm__ int32_t* selectedCounts = reinterpret_cast<__gm__ int32_t*>(workspace_ + tiling_->selectedCountsOffset);
        __gm__ uint8_t* suppressed = workspace_ + tiling_->suppressedOffset +
                                     static_cast<uint64_t>(coreIdx_) * static_cast<uint64_t>(tiling_->numBoxes);

        const int64_t taskCount = static_cast<int64_t>(tiling_->batchSize) * static_cast<int64_t>(tiling_->numClasses);
        const uint32_t selectThreadNum = GetSimtThreadNum(tiling_->numBoxes);
        for (int64_t taskIdx = coreIdx_; taskIdx < taskCount; taskIdx += tiling_->usedCoreNum) {
            const int32_t currentTaskIdx = static_cast<int32_t>(taskIdx);
            const int32_t batchIdx = currentTaskIdx / tiling_->numClasses;
            const int32_t classIdx = currentTaskIdx - batchIdx * tiling_->numClasses;
            if (useHotUb_) {
                ProcessHotUbTask(selectedScores, selectedIndices, selectedCounts, reduceScores, reduceIndices,
                                 selectThreadNum, batchIdx, classIdx, currentTaskIdx);
            } else {
                // Very large inputs do not fit in UB as a whole. They use the
                // explicitly reserved SIMT data cache while retaining the same
                // bounded UB reduction scratch.
                asc_vf_call<SelectClassNms>(dim3(selectThreadNum), boxes_, scores_, selectedScores, selectedIndices,
                                            selectedCounts, suppressed, reduceScores, reduceIndices, batchIdx, classIdx,
                                            currentTaskIdx, tiling_->numBoxes, tiling_->boxClasses, tiling_->numClasses,
                                            tiling_->maxOutputPerClass, tiling_->iouThreshold, tiling_->scoreThreshold);
            }
        }

        SyncAll();
        asc_dcci_entire_out();
        ProcessMerge(selectedScores, selectedIndices, selectedCounts, reduceScores, reduceIndices);
    }

private:
    __aicore__ inline void WaitVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(eventVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3_);
    }

    __aicore__ inline void WaitMte3ToV()
    {
        // The next consumer of these shared UB buffers is another SIMT VF.
        SetFlag<HardEvent::MTE3_V>(eventMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventMte3ToV_);
    }

    __aicore__ inline void ProcessHotUbTask(__gm__ float* selectedScores, __gm__ int32_t* selectedIndices,
                                            __gm__ int32_t* selectedCounts, __ubuf__ float* reduceScores,
                                            __ubuf__ int32_t* reduceIndices, uint32_t selectThreadNum, int32_t batchIdx,
                                            int32_t classIdx, int32_t taskIdx)
    {
        LocalTensor<float> boxesLocal = hotBoxesBuffer_.Get<float>();
        LocalTensor<float> scoresLocal = hotScoresBuffer_.Get<float>();
        LocalTensor<uint8_t> suppressedLocal = hotSuppressedBuffer_.Get<uint8_t>();
        LocalTensor<float> selectedScoresLocal = selectedScoresBuffer_.Get<float>();
        LocalTensor<int32_t> selectedIndicesLocal = selectedIndicesBuffer_.Get<int32_t>();
        LocalTensor<int32_t> selectedCountLocal = selectedCountBuffer_.Get<int32_t>();

        asc_vf_call<LoadTaskHotData>(dim3(selectThreadNum), boxes_, scores_,
                                     reinterpret_cast<__ubuf__ float*>(boxesLocal.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(scoresLocal.GetPhyAddr()), batchIdx, classIdx,
                                     tiling_->numBoxes, tiling_->boxClasses, tiling_->numClasses);
        asc_vf_call<SelectClassNmsUb>(dim3(selectThreadNum), reinterpret_cast<__ubuf__ float*>(boxesLocal.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float*>(scoresLocal.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float*>(selectedScoresLocal.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ int32_t*>(selectedIndicesLocal.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ int32_t*>(selectedCountLocal.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ uint8_t*>(suppressedLocal.GetPhyAddr()), reduceScores,
                                      reduceIndices, tiling_->numBoxes, tiling_->maxOutputPerClass,
                                      tiling_->iouThreshold, tiling_->scoreThreshold);

        GlobalTensor<float> selectedScoresGm;
        GlobalTensor<int32_t> selectedIndicesGm;
        GlobalTensor<int32_t> selectedCountsGm;
        selectedScoresGm.SetGlobalBuffer(selectedScores);
        selectedIndicesGm.SetGlobalBuffer(selectedIndices);
        selectedCountsGm.SetGlobalBuffer(selectedCounts);
        const int32_t selectedBase = taskIdx * tiling_->maxOutputPerClass;
        const DataCopyExtParams scoreCopyParams{1, static_cast<uint32_t>(tiling_->maxOutputPerClass * sizeof(float)), 0,
                                                0, 0};
        const DataCopyExtParams indexCopyParams{1, static_cast<uint32_t>(tiling_->maxOutputPerClass * sizeof(int32_t)),
                                                0, 0, 0};
        const DataCopyExtParams countCopyParams{1, sizeof(int32_t), 0, 0, 0};
        WaitVToMte3();
        DataCopyPad(selectedScoresGm[selectedBase], selectedScoresLocal, scoreCopyParams);
        DataCopyPad(selectedIndicesGm[selectedBase], selectedIndicesLocal, indexCopyParams);
        DataCopyPad(selectedCountsGm[taskIdx], selectedCountLocal, countCopyParams);
        WaitMte3ToV();
    }

    __aicore__ inline void ProcessMerge(__gm__ float* selectedScores, __gm__ int32_t* selectedIndices,
                                        __gm__ int32_t* selectedCounts, __ubuf__ float* reduceScores,
                                        __ubuf__ int32_t* reduceIndices)
    {
        LocalTensor<uint8_t> mergeState = mergeStateBuffer_.Get<uint8_t>();
        __ubuf__ int32_t* classCursors = reinterpret_cast<__ubuf__ int32_t*>(mergeState.GetPhyAddr());
        __ubuf__ int32_t* classCounts = classCursors + MAX_NUM_CLASSES;
        __ubuf__ float* headScores = reinterpret_cast<__ubuf__ float*>(classCounts + MAX_NUM_CLASSES);
        __ubuf__ int32_t* headIndices = reinterpret_cast<__ubuf__ int32_t*>(headScores + MAX_NUM_CLASSES);
        LocalTensor<float> outputBoxesLocal = outputBoxesBuffer_.Get<float>();
        LocalTensor<float> outputScoresLocal = outputScoresBuffer_.Get<float>();
        LocalTensor<float> outputClassesLocal = outputClassesBuffer_.Get<float>();
        LocalTensor<int32_t> validDetectionsLocal = validDetectionsBuffer_.Get<int32_t>();

        GlobalTensor<float> nmsedBoxesGm;
        GlobalTensor<float> nmsedScoresGm;
        GlobalTensor<float> nmsedClassesGm;
        GlobalTensor<int32_t> validDetectionsGm;
        nmsedBoxesGm.SetGlobalBuffer(nmsedBoxes_);
        nmsedScoresGm.SetGlobalBuffer(nmsedScores_);
        nmsedClassesGm.SetGlobalBuffer(nmsedClasses_);
        validDetectionsGm.SetGlobalBuffer(validDetections_);

        const uint32_t mergeThreadNum = GetSimtThreadNum(tiling_->numClasses);
        for (int32_t batchIdx = coreIdx_; batchIdx < tiling_->batchSize; batchIdx += tiling_->usedCoreNum) {
            asc_vf_call<MergeBatchResults>(
                dim3(mergeThreadNum), boxes_, selectedScores, selectedIndices, selectedCounts,
                reinterpret_cast<__ubuf__ float*>(outputBoxesLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(outputScoresLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(outputClassesLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(validDetectionsLocal.GetPhyAddr()), reduceScores, reduceIndices,
                classCursors, classCounts, headScores, headIndices, batchIdx, tiling_->numBoxes, tiling_->boxClasses,
                tiling_->numClasses, tiling_->maxOutputPerClass, tiling_->outputSize, tiling_->clipBoxes);

            const int64_t outputBase = static_cast<int64_t>(batchIdx) * tiling_->outputSize;
            const DataCopyExtParams boxesCopyParams{1, static_cast<uint32_t>(tiling_->outputSize * 4 * sizeof(float)),
                                                    0, 0, 0};
            const DataCopyExtParams vectorCopyParams{1, static_cast<uint32_t>(tiling_->outputSize * sizeof(float)), 0,
                                                     0, 0};
            const DataCopyExtParams validCopyParams{1, sizeof(int32_t), 0, 0, 0};
            WaitVToMte3();
            DataCopyPad(nmsedBoxesGm[outputBase * 4], outputBoxesLocal, boxesCopyParams);
            DataCopyPad(nmsedScoresGm[outputBase], outputScoresLocal, vectorCopyParams);
            DataCopyPad(nmsedClassesGm[outputBase], outputClassesLocal, vectorCopyParams);
            DataCopyPad(validDetectionsGm[batchIdx], validDetectionsLocal, validCopyParams);
            WaitMte3ToV();
        }
    }

    __gm__ float* boxes_ = nullptr;
    __gm__ float* scores_ = nullptr;
    __gm__ float* nmsedBoxes_ = nullptr;
    __gm__ float* nmsedScores_ = nullptr;
    __gm__ float* nmsedClasses_ = nullptr;
    __gm__ int32_t* validDetections_ = nullptr;
    __gm__ uint8_t* workspace_ = nullptr;
    const CombinedNonMaxSuppressionTilingData* tiling_ = nullptr;
    TPipe* pipe_ = nullptr;
    event_t eventVToMte3_;
    event_t eventMte3ToV_;
    int32_t coreIdx_ = 0;
    bool useHotUb_ = false;
    TBuf<TPosition::VECCALC> scratchBuffer_;
    TBuf<TPosition::VECCALC> hotBoxesBuffer_;
    TBuf<TPosition::VECCALC> hotScoresBuffer_;
    TBuf<TPosition::VECCALC> hotSuppressedBuffer_;
    TBuf<TPosition::VECCALC> selectedScoresBuffer_;
    TBuf<TPosition::VECCALC> selectedIndicesBuffer_;
    TBuf<TPosition::VECCALC> selectedCountBuffer_;
    TBuf<TPosition::VECCALC> mergeStateBuffer_;
    TBuf<TPosition::VECCALC> outputBoxesBuffer_;
    TBuf<TPosition::VECCALC> outputScoresBuffer_;
    TBuf<TPosition::VECCALC> outputClassesBuffer_;
    TBuf<TPosition::VECCALC> validDetectionsBuffer_;
};

} // namespace CombinedNonMaxSuppressionOps

#endif // COMBINED_NON_MAX_SUPPRESSION_SIMT_H_
