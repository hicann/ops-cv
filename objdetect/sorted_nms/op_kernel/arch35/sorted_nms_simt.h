/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORTED_NMS_SIMT_H_
#define SORTED_NMS_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "sorted_nms_tiling_data.h"

namespace NsSortedNMS {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t LOCAL_PAIRWISE_THREAD_NUM = 1024;
constexpr uint32_t MIN_THREAD_NUM = 32;
constexpr int32_t BOX_COORDS = 4;
constexpr int32_t INIT_INDEX = -1;
constexpr int32_t NOT_SUPPRESSED = 0;
constexpr int32_t SUPPRESSED = 1;
constexpr int32_t WORK_SELECTED_COUNT = 0;
constexpr int32_t WORK_CURRENT_INDEX = 1;
constexpr int32_t OUTPUT_SHAPE_INFO_SIZE = 9;
constexpr int32_t MASK_BITS = 32;
constexpr int64_t MULTI_CORE_MIN_BOXES = 1025;
constexpr int64_t MULTI_CORE_MAX_BOXES = 8192;
constexpr uint32_t UB_BLOCK_SIZE = 32;
// shape_out uses uint64_t entries. Bit 31 marks the rank field as uint64_t encoded.
constexpr uint64_t OUTPUT_SHAPE_RANK_ONE = 0x80000001ULL;

__aicore__ inline uint32_t AlignUbBytes(uint32_t bytes)
{
    return (bytes + UB_BLOCK_SIZE - 1U) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

__aicore__ inline uint32_t GetSimtThreadNum(int64_t workItems)
{
    uint32_t threadNum = MIN_THREAD_NUM;
    while (static_cast<int64_t>(threadNum) < workItems && threadNum < THREAD_NUM) {
        threadNum <<= 1;
    }
    return threadNum;
}

__simt_callee__ __aicore__ inline void WriteOutputShape(__gm__ uint64_t* outputShape, int32_t selectedCount)
{
    for (int32_t index = 0; index < OUTPUT_SHAPE_INFO_SIZE; ++index) {
        outputShape[index] = 0;
    }
    outputShape[0] = OUTPUT_SHAPE_RANK_ONE;
    outputShape[1] = static_cast<uint64_t>(selectedCount);
}

__simt_callee__ __aicore__ inline float ReadAsFloat(float val) { return val; }

__simt_callee__ __aicore__ inline float ReadAsFloat(half val) { return __half2float(val); }

template <typename TScore, typename TScoreThreshold>
__simt_vf__ __aicore__ inline void FindActiveBoxesNum(int64_t boxesNum, __gm__ TScore* sortedScores,
                                                      __gm__ TScoreThreshold* scoreThreshold,
                                                      __gm__ int32_t* activeBoxesNum)
{
    if (threadIdx.x != 0) {
        return;
    }
    const float scoreThr = ReadAsFloat(scoreThreshold[0]);
    int64_t left = 0;
    int64_t right = boxesNum;
    while (left < right) {
        const int64_t middle = left + (right - left) / 2;
        if (ReadAsFloat(sortedScores[middle]) > scoreThr) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    activeBoxesNum[0] = static_cast<int32_t>(left);
}

__simt_callee__ __aicore__ inline float ClampNonNegative(float val) { return val > 0.0f ? val : 0.0f; }

template <typename T>
__simt_callee__ __aicore__ inline float BoxArea(__gm__ T* boxes, int32_t boxIdx, float offset)
{
    int64_t base = static_cast<int64_t>(boxIdx) * BOX_COORDS;
    float x1 = ReadAsFloat(boxes[base]);
    float y1 = ReadAsFloat(boxes[base + 1]);
    float x2 = ReadAsFloat(boxes[base + 2]);
    float y2 = ReadAsFloat(boxes[base + 3]);
    float width = ClampNonNegative(x2 - x1 + offset);
    float height = ClampNonNegative(y2 - y1 + offset);
    return width * height;
}

template <typename T>
__simt_callee__ __aicore__ inline float Intersection(__gm__ T* boxes, int32_t lhs, int32_t rhs, float offset)
{
    int64_t lhsBase = static_cast<int64_t>(lhs) * BOX_COORDS;
    int64_t rhsBase = static_cast<int64_t>(rhs) * BOX_COORDS;
    float x1 = fmaxf(ReadAsFloat(boxes[lhsBase]), ReadAsFloat(boxes[rhsBase]));
    float y1 = fmaxf(ReadAsFloat(boxes[lhsBase + 1]), ReadAsFloat(boxes[rhsBase + 1]));
    float x2 = fminf(ReadAsFloat(boxes[lhsBase + 2]), ReadAsFloat(boxes[rhsBase + 2]));
    float y2 = fminf(ReadAsFloat(boxes[lhsBase + 3]), ReadAsFloat(boxes[rhsBase + 3]));
    float width = ClampNonNegative(x2 - x1 + offset);
    float height = ClampNonNegative(y2 - y1 + offset);
    return width * height;
}

template <typename T>
__simt_callee__ __aicore__ inline float Iou(__gm__ T* boxes, int32_t lhs, int32_t rhs, float offset)
{
    float inter = Intersection(boxes, lhs, rhs, offset);
    float lhsArea = BoxArea(boxes, lhs, offset);
    float rhsArea = BoxArea(boxes, rhs, offset);
    float denom = lhsArea + rhsArea - inter;
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return inter / denom;
}

template <typename T>
__simt_callee__ __aicore__ inline float IntersectionLocal(__ubuf__ T* boxes, int32_t lhs, int32_t rhs, float offset)
{
    int64_t lhsBase = static_cast<int64_t>(lhs) * BOX_COORDS;
    int64_t rhsBase = static_cast<int64_t>(rhs) * BOX_COORDS;
    float x1 = fmaxf(ReadAsFloat(boxes[lhsBase]), ReadAsFloat(boxes[rhsBase]));
    float y1 = fmaxf(ReadAsFloat(boxes[lhsBase + 1]), ReadAsFloat(boxes[rhsBase + 1]));
    float x2 = fminf(ReadAsFloat(boxes[lhsBase + 2]), ReadAsFloat(boxes[rhsBase + 2]));
    float y2 = fminf(ReadAsFloat(boxes[lhsBase + 3]), ReadAsFloat(boxes[rhsBase + 3]));
    float width = ClampNonNegative(x2 - x1 + offset);
    float height = ClampNonNegative(y2 - y1 + offset);
    return width * height;
}

template <typename TBox>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void BuildBoxAreas(int64_t boxesNum, int32_t offset,
                                                                               __ubuf__ TBox* boxes,
                                                                               __ubuf__ float* boxAreas)
{
    const float offsetVal = static_cast<float>(offset);
    for (int64_t boxIndex = threadIdx.x; boxIndex < boxesNum; boxIndex += blockDim.x) {
        const int64_t base = boxIndex * BOX_COORDS;
        const float width = ClampNonNegative(ReadAsFloat(boxes[base + 2]) - ReadAsFloat(boxes[base]) + offsetVal);
        const float height = ClampNonNegative(ReadAsFloat(boxes[base + 3]) - ReadAsFloat(boxes[base + 1]) + offsetVal);
        boxAreas[boxIndex] = width * height;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline float IouLocal(__ubuf__ T* boxes, __ubuf__ float* boxAreas, int32_t lhs, int32_t rhs,
                                                 float offset)
{
    const float inter = IntersectionLocal(boxes, lhs, rhs, offset);
    const float denom = boxAreas[lhs] + boxAreas[rhs] - inter;
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return inter / denom;
}

template <typename TBox, typename TScore, typename TIouThreshold, typename TScoreThreshold>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void SortedNMSSingleCore(
    int64_t boxesNum, int32_t offset, __gm__ TBox* boxes, __gm__ TScore* sortedScores, __gm__ int32_t* inputIndices,
    __gm__ int32_t* maxOutputSize, __gm__ TIouThreshold* iouThreshold, __gm__ TScoreThreshold* scoreThreshold,
    __gm__ int32_t* selectedIndices, __gm__ uint64_t* outputShape, __gm__ int32_t* work)
{
    if (boxesNum <= 0) {
        if (threadIdx.x == 0) {
            WriteOutputShape(outputShape, 0);
        }
        return;
    }

    __gm__ int32_t* control = work;
    __gm__ int32_t* suppressed = work + 2;

    for (int64_t idx = threadIdx.x; idx < boxesNum; idx += blockDim.x) {
        suppressed[idx] = NOT_SUPPRESSED;
    }
    if (threadIdx.x == 0) {
        control[WORK_SELECTED_COUNT] = 0;
        control[WORK_CURRENT_INDEX] = INIT_INDEX;
    }
    asc_threadfence_block();
    asc_syncthreads();

    int32_t maxOut = maxOutputSize[0];
    if (maxOut < 0) {
        maxOut = 0;
    }
    if (static_cast<int64_t>(maxOut) > boxesNum) {
        maxOut = static_cast<int32_t>(boxesNum);
    }
    float iouThr = ReadAsFloat(iouThreshold[0]);
    float scoreThr = ReadAsFloat(scoreThreshold[0]);
    float offsetVal = static_cast<float>(offset);

    for (int64_t sortedPos = 0; sortedPos < boxesNum; ++sortedPos) {
        if (threadIdx.x == 0) {
            control[WORK_CURRENT_INDEX] = INIT_INDEX;
            int32_t selectedCount = control[WORK_SELECTED_COUNT];
            if (selectedCount < maxOut && suppressed[sortedPos] == NOT_SUPPRESSED) {
                float score = ReadAsFloat(sortedScores[sortedPos]);
                int32_t current = inputIndices[sortedPos];
                if (score > scoreThr && current >= 0 && static_cast<int64_t>(current) < boxesNum) {
                    selectedIndices[selectedCount] = current;
                    control[WORK_SELECTED_COUNT] = selectedCount + 1;
                    control[WORK_CURRENT_INDEX] = current;
                }
            }
        }
        asc_threadfence_block();
        asc_syncthreads();

        int32_t currentIndex = control[WORK_CURRENT_INDEX];
        if (currentIndex >= 0) {
            for (int64_t nextPos = sortedPos + 1 + threadIdx.x; nextPos < boxesNum; nextPos += blockDim.x) {
                if (suppressed[nextPos] == NOT_SUPPRESSED) {
                    float score = ReadAsFloat(sortedScores[nextPos]);
                    int32_t nextIndex = inputIndices[nextPos];
                    if (score > scoreThr && nextIndex >= 0 && static_cast<int64_t>(nextIndex) < boxesNum) {
                        float overlap = Iou(boxes, currentIndex, nextIndex, offsetVal);
                        if (overlap > iouThr) {
                            suppressed[nextPos] = SUPPRESSED;
                        }
                    }
                }
            }
        }
        asc_threadfence_block();
        asc_syncthreads();

        if (control[WORK_SELECTED_COUNT] >= maxOut) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        WriteOutputShape(outputShape, control[WORK_SELECTED_COUNT]);
    }
}

template <typename TBox, typename TScore, typename TIouThreshold, typename TScoreThreshold>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void BuildPairwiseMasksLocal(
    int64_t boxesNum, int64_t activeBoxesNum, int64_t activeMaskWordNum, int32_t offset, int32_t coreIdx,
    int32_t coreNum, __ubuf__ TBox* boxes, __gm__ TScore* sortedScores, __gm__ int32_t* inputIndices,
    __ubuf__ float* boxAreas, __gm__ TIouThreshold* iouThreshold, __gm__ TScoreThreshold* scoreThreshold,
    __gm__ uint32_t* pairwiseMasks)
{
    float iouThr = ReadAsFloat(iouThreshold[0]);
    float offsetVal = static_cast<float>(offset);
    const int64_t globalThreadIdx = static_cast<int64_t>(coreIdx) * blockDim.x + threadIdx.x;
    const int64_t globalThreadNum = static_cast<int64_t>(coreNum) * blockDim.x;
    const int64_t pairwiseMaskWords = activeBoxesNum * activeMaskWordNum;

    for (int64_t task = globalThreadIdx; task < pairwiseMaskWords; task += globalThreadNum) {
        const int64_t sortedPos = task / activeMaskWordNum;
        const int64_t wordIndex = task - sortedPos * activeMaskWordNum;
        const int64_t firstRelevantWord = (sortedPos + 1) / MASK_BITS;
        if (wordIndex < firstRelevantWord) {
            continue;
        }
        const int64_t firstNextPos = wordIndex * MASK_BITS;
        uint32_t mask = 0;
        const int32_t currentIndex = inputIndices[sortedPos];
        if (currentIndex >= 0 && static_cast<int64_t>(currentIndex) < boxesNum) {
            for (int32_t bit = 0; bit < MASK_BITS; ++bit) {
                const int64_t nextPos = firstNextPos + bit;
                if (nextPos <= sortedPos || nextPos >= activeBoxesNum) {
                    continue;
                }
                const int32_t nextIndex = inputIndices[nextPos];
                if (nextIndex >= 0 && static_cast<int64_t>(nextIndex) < boxesNum &&
                    IouLocal(boxes, boxAreas, currentIndex, nextIndex, offsetVal) > iouThr) {
                    mask |= 1U << bit;
                }
            }
        }
        pairwiseMasks[task] = mask;
    }
}

template <typename TBox, typename TScore, typename TIouThreshold, typename TScoreThreshold>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void BuildPairwiseMasks(
    int64_t boxesNum, int64_t activeBoxesNum, int64_t activeMaskWordNum, int32_t offset, int32_t coreIdx,
    int32_t coreNum, __gm__ TBox* boxes, __gm__ TScore* sortedScores, __gm__ int32_t* inputIndices,
    __gm__ TIouThreshold* iouThreshold, __gm__ TScoreThreshold* scoreThreshold, __gm__ uint32_t* pairwiseMasks)
{
    float iouThr = ReadAsFloat(iouThreshold[0]);
    float offsetVal = static_cast<float>(offset);
    const int64_t globalThreadIdx = static_cast<int64_t>(coreIdx) * blockDim.x + threadIdx.x;
    const int64_t globalThreadNum = static_cast<int64_t>(coreNum) * blockDim.x;
    const int64_t pairwiseMaskWords = activeBoxesNum * activeMaskWordNum;

    for (int64_t task = globalThreadIdx; task < pairwiseMaskWords; task += globalThreadNum) {
        const int64_t sortedPos = task / activeMaskWordNum;
        const int64_t wordIndex = task - sortedPos * activeMaskWordNum;
        const int64_t firstRelevantWord = (sortedPos + 1) / MASK_BITS;
        if (wordIndex < firstRelevantWord) {
            continue;
        }
        const int64_t firstNextPos = wordIndex * MASK_BITS;
        uint32_t mask = 0;
        const int32_t currentIndex = inputIndices[sortedPos];
        if (currentIndex >= 0 && static_cast<int64_t>(currentIndex) < boxesNum) {
            for (int32_t bit = 0; bit < MASK_BITS; ++bit) {
                const int64_t nextPos = firstNextPos + bit;
                if (nextPos <= sortedPos || nextPos >= activeBoxesNum) {
                    continue;
                }
                const int32_t nextIndex = inputIndices[nextPos];
                if (nextIndex >= 0 && static_cast<int64_t>(nextIndex) < boxesNum &&
                    Iou(boxes, currentIndex, nextIndex, offsetVal) > iouThr) {
                    mask |= 1U << bit;
                }
            }
        }
        pairwiseMasks[task] = mask;
    }
}

template <typename TScore, typename TScoreThreshold>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void SelectFromPairwiseMasks(
    int64_t boxesNum, int64_t activeBoxesNum, int64_t activeMaskWordNum, __gm__ TScore* sortedScores,
    __gm__ int32_t* inputIndices, __gm__ int32_t* maxOutputSize, __gm__ TScoreThreshold* scoreThreshold,
    __gm__ int32_t* selectedIndices, __gm__ uint64_t* outputShape, __gm__ int32_t* work)
{
    __gm__ int32_t* control = work;
    __gm__ uint32_t* suppressedMasks = (__gm__ uint32_t*)(work + 2);
    __gm__ uint32_t* pairwiseMasks = suppressedMasks + activeMaskWordNum;

    for (int64_t wordIndex = threadIdx.x; wordIndex < activeMaskWordNum; wordIndex += blockDim.x) {
        suppressedMasks[wordIndex] = 0;
    }
    if (threadIdx.x == 0) {
        control[WORK_SELECTED_COUNT] = 0;
        control[WORK_CURRENT_INDEX] = INIT_INDEX;
    }
    asc_threadfence_block();
    asc_syncthreads();

    int32_t maxOut = maxOutputSize[0];
    maxOut = maxOut < 0 ? 0 : maxOut;
    if (static_cast<int64_t>(maxOut) > boxesNum) {
        maxOut = static_cast<int32_t>(boxesNum);
    }
    if (maxOut == 0 || activeBoxesNum == 0) {
        if (threadIdx.x == 0) {
            WriteOutputShape(outputShape, 0);
        }
        return;
    }

    for (int64_t sortedPos = 0; sortedPos < activeBoxesNum; ++sortedPos) {
        if (threadIdx.x == 0) {
            control[WORK_CURRENT_INDEX] = INIT_INDEX;
            const int64_t wordIndex = sortedPos / MASK_BITS;
            const int32_t bitIndex = static_cast<int32_t>(sortedPos - wordIndex * MASK_BITS);
            const bool isSuppressed = (suppressedMasks[wordIndex] & (1U << bitIndex)) != 0;
            const int32_t currentIndex = inputIndices[sortedPos];
            if (!isSuppressed && currentIndex >= 0 && static_cast<int64_t>(currentIndex) < boxesNum) {
                const int32_t selectedCount = control[WORK_SELECTED_COUNT];
                selectedIndices[selectedCount] = currentIndex;
                control[WORK_SELECTED_COUNT] = selectedCount + 1;
                control[WORK_CURRENT_INDEX] = static_cast<int32_t>(sortedPos);
            }
        }
        asc_threadfence_block();
        asc_syncthreads();

        const int32_t selectedRow = control[WORK_CURRENT_INDEX];
        if (selectedRow >= 0) {
            const int64_t rowOffset = static_cast<int64_t>(selectedRow) * activeMaskWordNum;
            const int64_t firstRelevantWord = (static_cast<int64_t>(selectedRow) + 1) / MASK_BITS;
            for (int64_t wordIndex = firstRelevantWord + threadIdx.x; wordIndex < activeMaskWordNum;
                 wordIndex += blockDim.x) {
                suppressedMasks[wordIndex] |= pairwiseMasks[rowOffset + wordIndex];
            }
        }
        asc_threadfence_block();
        asc_syncthreads();

        if (control[WORK_SELECTED_COUNT] >= maxOut) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        WriteOutputShape(outputShape, control[WORK_SELECTED_COUNT]);
    }
}

template <typename TBox, typename TScore, typename TIouThreshold, typename TScoreThreshold>
__aicore__ inline void Process(GM_ADDR boxes, GM_ADDR sortedScores, GM_ADDR inputIndices, GM_ADDR maxOutputSize,
                               GM_ADDR iouThreshold, GM_ADDR scoreThreshold, GM_ADDR selectedIndices,
                               GM_ADDR outputShape, GM_ADDR workspace, const SortedNMSTilingData* tilingData,
                               TPipe* pipe)
{
    __gm__ TBox* boxesGm = (__gm__ TBox*)boxes;
    __gm__ TScore* sortedScoresGm = (__gm__ TScore*)sortedScores;
    __gm__ int32_t* inputIndicesGm = (__gm__ int32_t*)inputIndices;
    __gm__ int32_t* maxOutputSizeGm = (__gm__ int32_t*)maxOutputSize;
    __gm__ TIouThreshold* iouThresholdGm = (__gm__ TIouThreshold*)iouThreshold;
    __gm__ TScoreThreshold* scoreThresholdGm = (__gm__ TScoreThreshold*)scoreThreshold;
    __gm__ int32_t* selectedIndicesGm = (__gm__ int32_t*)selectedIndices;
    __gm__ uint64_t* outputShapeGm = (__gm__ uint64_t*)outputShape;
    __gm__ int32_t* workGm = (__gm__ int32_t*)workspace;
    const uint32_t singleCoreThreadNum = GetSimtThreadNum(tilingData->boxesNum);

    if (tilingData->boxesNum >= MULTI_CORE_MIN_BOXES && tilingData->boxesNum <= MULTI_CORE_MAX_BOXES) {
        const int32_t coreIdx = static_cast<int32_t>(GetBlockIdx());
        const int64_t maskWordNum = (tilingData->boxesNum + MASK_BITS - 1) / MASK_BITS;
        int32_t maxOut = maxOutputSizeGm[0];
        maxOut = maxOut < 0 ? 0 : maxOut;
        if (static_cast<int64_t>(maxOut) > tilingData->boxesNum) {
            maxOut = static_cast<int32_t>(tilingData->boxesNum);
        }
        const int64_t pairwiseBreakEven = (tilingData->boxesNum + tilingData->coreNum - 1) / tilingData->coreNum;
        if (static_cast<int64_t>(maxOut) <= pairwiseBreakEven) {
            if (coreIdx == 0) {
                asc_vf_call<SortedNMSSingleCore<TBox, TScore, TIouThreshold, TScoreThreshold>>(
                    dim3(singleCoreThreadNum), tilingData->boxesNum, tilingData->offset, boxesGm, sortedScoresGm,
                    inputIndicesGm, maxOutputSizeGm, iouThresholdGm, scoreThresholdGm, selectedIndicesGm, outputShapeGm,
                    workGm);
            }
            return;
        }

        __gm__ uint32_t* suppressedMasks = (__gm__ uint32_t*)(workGm + 2);
        if (coreIdx == 0) {
            asc_vf_call<FindActiveBoxesNum<TScore, TScoreThreshold>>(dim3(1), tilingData->boxesNum, sortedScoresGm,
                                                                     scoreThresholdGm, workGm);
        }
        SyncAll();
        const int64_t activeBoxesNum = static_cast<int64_t>(workGm[0]);
        const int64_t activeMaskWordNum = (activeBoxesNum + MASK_BITS - 1) / MASK_BITS;
        __gm__ uint32_t* pairwiseMasks = suppressedMasks + activeMaskWordNum;
        if (tilingData->useLocalBoxes != 0) {
            const uint32_t boxesBytes = AlignUbBytes(
                static_cast<uint32_t>(tilingData->boxesNum * BOX_COORDS * sizeof(TBox)));
            const uint32_t areasBytes = AlignUbBytes(static_cast<uint32_t>(tilingData->boxesNum * sizeof(float)));
            TQue<QuePosition::VECIN, 1> boxesQueue;
            TBuf<TPosition::VECCALC> areasBuffer;
            pipe->InitBuffer(boxesQueue, 1, boxesBytes);
            pipe->InitBuffer(areasBuffer, areasBytes);

            GlobalTensor<TBox> boxesTensor;
            boxesTensor.SetGlobalBuffer(boxesGm, tilingData->boxesNum * BOX_COORDS);
            LocalTensor<TBox> boxesLocal = boxesQueue.AllocTensor<TBox>();
            const DataCopyExtParams boxesCopyParams{
                1, static_cast<uint32_t>(tilingData->boxesNum * BOX_COORDS * sizeof(TBox)), 0, 0, 0};
            DataCopyPad(boxesLocal, boxesTensor, boxesCopyParams, DataCopyPadExtParams<TBox>{false, 0, 0, 0});
            boxesQueue.EnQue(boxesLocal);
            boxesLocal = boxesQueue.DeQue<TBox>();
            LocalTensor<float> areasLocal = areasBuffer.Get<float>();
            __ubuf__ TBox* boxesUb = reinterpret_cast<__ubuf__ TBox*>(boxesLocal.GetPhyAddr());
            __ubuf__ float* areasUb = reinterpret_cast<__ubuf__ float*>(areasLocal.GetPhyAddr());

            asc_vf_call<BuildBoxAreas<TBox>>(dim3(GetSimtThreadNum(tilingData->boxesNum)), tilingData->boxesNum,
                                             tilingData->offset, boxesUb, areasUb);
            event_t areasReady = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(areasReady);
            WaitFlag<HardEvent::V_S>(areasReady);
            asc_vf_call<BuildPairwiseMasksLocal<TBox, TScore, TIouThreshold, TScoreThreshold>>(
                dim3(LOCAL_PAIRWISE_THREAD_NUM), tilingData->boxesNum, activeBoxesNum, activeMaskWordNum,
                tilingData->offset, coreIdx, tilingData->coreNum, boxesUb, sortedScoresGm, inputIndicesGm, areasUb,
                iouThresholdGm, scoreThresholdGm, pairwiseMasks);
            event_t pairwiseDone = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(pairwiseDone);
            WaitFlag<HardEvent::V_S>(pairwiseDone);
            boxesQueue.FreeTensor(boxesLocal);
        } else {
            asc_vf_call<BuildPairwiseMasks<TBox, TScore, TIouThreshold, TScoreThreshold>>(
                dim3(THREAD_NUM), tilingData->boxesNum, activeBoxesNum, activeMaskWordNum, tilingData->offset, coreIdx,
                tilingData->coreNum, boxesGm, sortedScoresGm, inputIndicesGm, iouThresholdGm, scoreThresholdGm,
                pairwiseMasks);
        }
        SyncAll();
        if (coreIdx == 0) {
            const uint32_t selectThreadNum = GetSimtThreadNum(activeMaskWordNum);
            asc_vf_call<SelectFromPairwiseMasks<TScore, TScoreThreshold>>(
                dim3(selectThreadNum), tilingData->boxesNum, activeBoxesNum, activeMaskWordNum, sortedScoresGm,
                inputIndicesGm, maxOutputSizeGm, scoreThresholdGm, selectedIndicesGm, outputShapeGm, workGm);
        }
        return;
    }

    asc_vf_call<SortedNMSSingleCore<TBox, TScore, TIouThreshold, TScoreThreshold>>(
        dim3(singleCoreThreadNum), tilingData->boxesNum, tilingData->offset, boxesGm, sortedScoresGm, inputIndicesGm,
        maxOutputSizeGm, iouThresholdGm, scoreThresholdGm, selectedIndicesGm, outputShapeGm, workGm);
}
} // namespace NsSortedNMS

#endif // SORTED_NMS_SIMT_H_
