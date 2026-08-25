/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NMS_V7_KERNEL_H_
#define NMS_V7_KERNEL_H_

#include <cstdint>

#include "c_api/cache_ctrl/cache_ctrl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "non_max_suppression_v7_tiling_data.h"

namespace NonMaxSuppressionV7Op {
using namespace AscendC;

namespace {
constexpr uint32_t kGatherThreadNum = 512;
constexpr float kNegativeInfinity = -(__builtin_inff());
constexpr float kMaxInt32AsFloat = 2147483520.0F;
constexpr float kMinInt32AsFloat = -2147483648.0F;
constexpr int32_t kMaxInt32 = 2147483647;
constexpr int32_t kMinInt32 = (-2147483647 - 1);
constexpr uint64_t kScratchFloatFieldCount = 6;
constexpr int64_t kScalarSlotElements = 8;
constexpr int64_t kOutputTileRows = 64;
constexpr int64_t kSmallBoxesThreshold = 16;

__aicore__ inline bool UseSerialPath(int64_t boxes, int64_t outputCapacity)
{
    return outputCapacity == 0 || boxes <= kSmallBoxesThreshold || (boxes <= 24 && outputCapacity <= 16) ||
           (boxes <= 40 && outputCapacity <= 4) || (boxes <= 64 && outputCapacity <= 2);
}

__aicore__ inline float MinFloat(float lhs, float rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline float MaxFloat(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }

template <typename T>
__aicore__ inline void LoadBox(const __gm__ T* boxes, int64_t offset, uint8_t centerPointBox, float& yMin, float& xMin,
                               float& yMax, float& xMax)
{
    const float first = static_cast<float>(boxes[offset]);
    const float second = static_cast<float>(boxes[offset + 1]);
    const float third = static_cast<float>(boxes[offset + 2]);
    const float fourth = static_cast<float>(boxes[offset + 3]);
    if (centerPointBox != 0) {
        const float y0 = second - fourth * 0.5F;
        const float y1 = second + fourth * 0.5F;
        const float x0 = first - third * 0.5F;
        const float x1 = first + third * 0.5F;
        yMin = MinFloat(y0, y1);
        yMax = MaxFloat(y0, y1);
        xMin = MinFloat(x0, x1);
        xMax = MaxFloat(x0, x1);
    } else {
        yMin = MinFloat(first, third);
        yMax = MaxFloat(first, third);
        xMin = MinFloat(second, fourth);
        xMax = MaxFloat(second, fourth);
    }
}

__aicore__ inline float ComputeIoU(float yMin, float xMin, float yMax, float xMax, float otherYMin, float otherXMin,
                                   float otherYMax, float otherXMax)
{
    const float intersectionHeight = MaxFloat(0.0F, MinFloat(yMax, otherYMax) - MaxFloat(yMin, otherYMin));
    const float intersectionWidth = MaxFloat(0.0F, MinFloat(xMax, otherXMax) - MaxFloat(xMin, otherXMin));
    const float intersection = intersectionHeight * intersectionWidth;
    const float area = MaxFloat(0.0F, yMax - yMin) * MaxFloat(0.0F, xMax - xMin);
    const float otherArea = MaxFloat(0.0F, otherYMax - otherYMin) * MaxFloat(0.0F, otherXMax - otherXMin);
    const float unionArea = area + otherArea - intersection;
    if (intersection <= 0.0F || unionArea <= 0.0F) {
        return 0.0F;
    }
    return intersection / unionArea;
}

__aicore__ inline int32_t ConvertIndexValue(float value)
{
    // Match CUDA's float16-to-int32 conversion used by the GPU competitor.
    // Finite float16 values truncate toward zero; NaN/+Inf/-Inf map to
    // 0/INT32_MAX/INT32_MIN respectively.
    if (value != value) {
        return 0;
    }
    if (value > kMaxInt32AsFloat) {
        return kMaxInt32;
    }
    if (value < kMinInt32AsFloat) {
        return kMinInt32;
    }
    return static_cast<int32_t>(value);
}
} // namespace

template <typename TBoxes, typename TScores>
__simt_vf__ __aicore__ __launch_bounds__(kGatherThreadNum) inline void GatherAllClassInputs(
    const __gm__ TBoxes* boxes, const __gm__ TScores* scores, __gm__ float* scratch,
    const __gm__ float* scoreThresholdInput, uint64_t taskStart, uint64_t taskStride, uint64_t taskCount,
    uint64_t boxesNum, uint64_t classesNum, uint64_t scratchFieldStride, uint8_t centerPointBox,
    uint8_t hasScoreThreshold, float defaultScoreThreshold)
{
    const float scoreThreshold = hasScoreThreshold != 0 ? scoreThresholdInput[0] : defaultScoreThreshold;
    const uint64_t fieldElements = scratchFieldStride / sizeof(float);
    for (uint64_t taskIndex = taskStart; taskIndex < taskCount; taskIndex += taskStride) {
        const uint64_t batchIndex = taskIndex / classesNum;
        __gm__ float* const taskScratch = scratch + taskIndex * fieldElements * kScratchFloatFieldCount;
        __gm__ float* const stageScores = taskScratch;
        __gm__ float* const stageYMin = stageScores + fieldElements;
        __gm__ float* const stageXMin = stageYMin + fieldElements;
        __gm__ float* const stageYMax = stageXMin + fieldElements;
        __gm__ float* const stageXMax = stageYMax + fieldElements;
        __gm__ float* const stageArea = stageXMax + fieldElements;
        for (uint64_t boxIndex = static_cast<uint64_t>(threadIdx.x); boxIndex < boxesNum;
             boxIndex += static_cast<uint64_t>(blockDim.x)) {
            const uint64_t boxOffset = (batchIndex * boxesNum + boxIndex) * 4;
            const float first = static_cast<float>(boxes[boxOffset]);
            const float second = static_cast<float>(boxes[boxOffset + 1]);
            const float third = static_cast<float>(boxes[boxOffset + 2]);
            const float fourth = static_cast<float>(boxes[boxOffset + 3]);
            float yMin;
            float xMin;
            float yMax;
            float xMax;
            if (centerPointBox != 0) {
                const float y0 = second - fourth * 0.5F;
                const float y1 = second + fourth * 0.5F;
                const float x0 = first - third * 0.5F;
                const float x1 = first + third * 0.5F;
                yMin = y0 < y1 ? y0 : y1;
                yMax = y0 > y1 ? y0 : y1;
                xMin = x0 < x1 ? x0 : x1;
                xMax = x0 > x1 ? x0 : x1;
            } else {
                yMin = first < third ? first : third;
                yMax = first > third ? first : third;
                xMin = second < fourth ? second : fourth;
                xMax = second > fourth ? second : fourth;
            }
            const float height = yMax > yMin ? yMax - yMin : 0.0F;
            const float width = xMax > xMin ? xMax - xMin : 0.0F;
            const float candidateScore = static_cast<float>(scores[taskIndex * boxesNum + boxIndex]);
            stageScores[boxIndex] = candidateScore > scoreThreshold ? candidateScore : kNegativeInfinity;
            stageYMin[boxIndex] = yMin;
            stageXMin[boxIndex] = xMin;
            stageYMax[boxIndex] = yMax;
            stageXMax[boxIndex] = xMax;
            stageArea[boxIndex] = height * width;
        }
    }
}

template <typename TBoxes, typename TScores>
class Kernel {
public:
    __aicore__ void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR maxOutput, GM_ADDR iou, GM_ADDR score, GM_ADDR index,
                         GM_ADDR out, GM_ADDR workspace, const NonMaxSuppressionV7TilingData* tiling)
    {
        boxes_ = reinterpret_cast<__gm__ TBoxes*>(boxes);
        scores_ = reinterpret_cast<__gm__ TScores*>(scores);
        maxOutput_ = reinterpret_cast<__gm__ int32_t*>(maxOutput);
        iou_ = reinterpret_cast<__gm__ float*>(iou);
        score_ = reinterpret_cast<__gm__ float*>(score);
        index_ = reinterpret_cast<__gm__ half*>(index);
        out_ = reinterpret_cast<__gm__ int32_t*>(out);
        tiling_ = tiling;
        if (tiling_->maxOutputSize == 0) {
            workspaceReady_ = true;
            return;
        }
        userWorkspace_ = GetUserWorkspace(workspace);
        workspaceReady_ = userWorkspace_ != nullptr;
        if (!workspaceReady_) {
            return;
        }
        scratch_ = reinterpret_cast<__gm__ float*>(userWorkspace_);
        classIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace_ + tiling_->classIndicesOffset),
                                        tiling_->batch * tiling_->classes * tiling_->maxOutputPerClass);
        classCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace_ + tiling_->classCountsOffset),
                                       tiling_->batch * tiling_->classes);
        outGm_.SetGlobalBuffer(out_, tiling_->maxOutputSize * 3);
        selectedBoxes_ = reinterpret_cast<__gm__ int32_t*>(userWorkspace_ + tiling_->classIndicesOffset);
        if (!UseSerialPath(tiling_->boxes, tiling_->maxOutputSize)) {
            InitTileBuffers();
        }
    }

    __aicore__ void Process()
    {
        const int64_t blockIndex = static_cast<int64_t>(GetBlockIdx());
        if (blockIndex >= tiling_->usedCoreNum || tiling_->maxOutputSize == 0) {
            return;
        }
        if (!workspaceReady_) {
            return;
        }
        const int64_t taskCount = tiling_->batch * tiling_->classes;
        float iouThreshold;
        float scoreThreshold;
        int64_t maxOutputPerClass;
        LoadRuntimeParameters(iouThreshold, scoreThreshold, maxOutputPerClass);
        if (UseSerialPath(tiling_->boxes, tiling_->maxOutputSize)) {
            if (blockIndex == 0) {
                ProcessSmall(maxOutputPerClass, iouThreshold, scoreThreshold);
            }
            return;
        }

        asc_vf_call<GatherAllClassInputs<TBoxes, TScores>>(
            dim3{kGatherThreadNum}, boxes_, scores_, scratch_, score_, static_cast<uint64_t>(blockIndex),
            static_cast<uint64_t>(tiling_->usedCoreNum), static_cast<uint64_t>(taskCount),
            static_cast<uint64_t>(tiling_->boxes), static_cast<uint64_t>(tiling_->classes), tiling_->scratchFieldStride,
            tiling_->centerPointBox, tiling_->hasScore, scoreThreshold);
        // The following vector stage reads the SIMT-produced GM scratch via DMA.
        // Publish every core's data-cache writes before the collective barrier.
        asc_dcci_entire_out();
        SyncAll();

        for (int64_t taskBase = 0; taskBase < taskCount; taskBase += tiling_->usedCoreNum) {
            const int64_t taskIndex = taskBase + blockIndex;
            if (taskIndex < taskCount) {
                ProcessClass(taskIndex, maxOutputPerClass, iouThreshold, scoreThreshold);
            }
            // Keep task publication collective, including the final partial
            // wave, before any core advances to the next workspace region.
            SyncAll();
        }

        if (blockIndex == 0) {
            MergeOutput(taskCount);
        }
    }

private:
    __aicore__ static int64_t MinInt64(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ void LoadRuntimeParameters(float& iouThreshold, float& scoreThreshold, int64_t& maxOutputPerClass)
    {
        iouThreshold = tiling_->hasIou != 0 ? iou_[0] : tiling_->iouThreshold;
        scoreThreshold = tiling_->hasScore != 0 ? score_[0] : tiling_->scoreThreshold;
        maxOutputPerClass = tiling_->maxOutputPerClass;
        if (tiling_->hasMax != 0) {
            const int64_t requested = static_cast<int64_t>(maxOutput_[0]);
            if (requested <= 0) {
                maxOutputPerClass = 0;
            } else if (requested < maxOutputPerClass) {
                maxOutputPerClass = requested;
            }
        }
    }

    __aicore__ bool IsSelectedSmall(int64_t classStart, int64_t written, int64_t boxIndex) const
    {
        for (int64_t selected = classStart; selected < written; ++selected) {
            if (selectedBoxes_[selected] == boxIndex) {
                return true;
            }
        }
        return false;
    }

    __aicore__ bool IsSuppressedSmall(int64_t batchIndex, int64_t candidateBox, int64_t classStart, int64_t written,
                                      float iouThreshold) const
    {
        const int64_t batchOffset = batchIndex * tiling_->boxes * 4;
        float candidateYMin;
        float candidateXMin;
        float candidateYMax;
        float candidateXMax;
        LoadBox(boxes_, batchOffset + candidateBox * 4, tiling_->centerPointBox, candidateYMin, candidateXMin,
                candidateYMax, candidateXMax);
        for (int64_t selected = classStart; selected < written; ++selected) {
            const int64_t selectedBox = selectedBoxes_[selected];
            if (selectedBox == candidateBox) {
                return true;
            }
            float selectedYMin;
            float selectedXMin;
            float selectedYMax;
            float selectedXMax;
            LoadBox(boxes_, batchOffset + selectedBox * 4, tiling_->centerPointBox, selectedYMin, selectedXMin,
                    selectedYMax, selectedXMax);
            if (ComputeIoU(candidateYMin, candidateXMin, candidateYMax, candidateXMax, selectedYMin, selectedXMin,
                           selectedYMax, selectedXMax) > iouThreshold) {
                return true;
            }
        }
        return false;
    }

    __aicore__ void ProcessSmall(int64_t maxOutputPerClass, float iouThreshold, float scoreThreshold)
    {
        int64_t written = 0;
        for (int64_t batchIndex = 0; batchIndex < tiling_->batch && written < tiling_->maxOutputSize; ++batchIndex) {
            for (int64_t classIndex = 0; classIndex < tiling_->classes && written < tiling_->maxOutputSize;
                 ++classIndex) {
                const int64_t classStart = written;
                int64_t selectedInClass = 0;
                while (selectedInClass < maxOutputPerClass && written < tiling_->maxOutputSize) {
                    float bestScore = kNegativeInfinity;
                    int64_t bestBox = -1;
                    for (int64_t boxIndex = 0; boxIndex < tiling_->boxes; ++boxIndex) {
                        const int64_t scoreOffset = (batchIndex * tiling_->classes + classIndex) * tiling_->boxes +
                                                    boxIndex;
                        const float candidateScore = static_cast<float>(scores_[scoreOffset]);
                        if (!(candidateScore > scoreThreshold) || IsSelectedSmall(classStart, written, boxIndex) ||
                            IsSuppressedSmall(batchIndex, boxIndex, classStart, written, iouThreshold)) {
                            continue;
                        }
                        if (bestBox < 0 || candidateScore > bestScore) {
                            bestScore = candidateScore;
                            bestBox = boxIndex;
                        }
                    }
                    if (bestBox < 0) {
                        break;
                    }
                    selectedBoxes_[written] = static_cast<int32_t>(bestBox);
                    WriteOutputSmall(batchIndex, classIndex, bestBox, written);
                    ++written;
                    ++selectedInClass;
                }
            }
        }
        while (written < tiling_->maxOutputSize) {
            out_[written * 3] = -1;
            out_[written * 3 + 1] = -1;
            out_[written * 3 + 2] = -1;
            ++written;
        }
    }

    __aicore__ void InitTileBuffers()
    {
        const int64_t tileBytes = tiling_->tileSize * static_cast<int64_t>(sizeof(float));
        pipe_.InitBuffer(scoreBuffer_, tileBytes);
        pipe_.InitBuffer(yMinBuffer_, tileBytes);
        pipe_.InitBuffer(xMinBuffer_, tileBytes);
        pipe_.InitBuffer(yMaxBuffer_, tileBytes);
        pipe_.InitBuffer(xMaxBuffer_, tileBytes);
        pipe_.InitBuffer(areaBuffer_, tileBytes);
        pipe_.InitBuffer(temp0Buffer_, tileBytes);
        pipe_.InitBuffer(temp1Buffer_, tileBytes);
        pipe_.InitBuffer(temp2Buffer_, tileBytes);
        pipe_.InitBuffer(temp3Buffer_, tileBytes);
        pipe_.InitBuffer(reduceWorkBuffer_, tiling_->reduceBufferSize * static_cast<int64_t>(sizeof(float)));
        pipe_.InitBuffer(reduceOutputBuffer_, 64);
        pipe_.InitBuffer(compareMaskBuffer_, tiling_->tileSize * static_cast<int64_t>(sizeof(uint8_t)));
        pipe_.InitBuffer(intScalarBuffer_, kOutputTileRows * 3 * static_cast<int64_t>(sizeof(int32_t)));
        pipe_.InitBuffer(selectedBoxBuffer_, 5 * kScalarSlotElements * static_cast<int64_t>(sizeof(float)));
        scoreLocal_ = scoreBuffer_.Get<float>();
        yMinLocal_ = yMinBuffer_.Get<float>();
        xMinLocal_ = xMinBuffer_.Get<float>();
        yMaxLocal_ = yMaxBuffer_.Get<float>();
        xMaxLocal_ = xMaxBuffer_.Get<float>();
        areaLocal_ = areaBuffer_.Get<float>();
        temp0Local_ = temp0Buffer_.Get<float>();
        temp1Local_ = temp1Buffer_.Get<float>();
        temp2Local_ = temp2Buffer_.Get<float>();
        temp3Local_ = temp3Buffer_.Get<float>();
        reduceWorkLocal_ = reduceWorkBuffer_.Get<float>();
        reduceOutputLocal_ = reduceOutputBuffer_.Get<float>();
        compareMaskLocal_ = compareMaskBuffer_.Get<uint8_t>();
        intScalarLocal_ = intScalarBuffer_.Get<int32_t>();
        selectedBoxLocal_ = selectedBoxBuffer_.Get<float>();
    }

    __aicore__ void SetTaskWorkspace(int64_t taskIndex)
    {
        const uint64_t fieldElements = tiling_->scratchFieldStride / sizeof(float);
        __gm__ float* taskScratch = scratch_ +
                                    static_cast<uint64_t>(taskIndex) * fieldElements * kScratchFloatFieldCount;
        stageScores_.SetGlobalBuffer(taskScratch, tiling_->boxes);
        taskScratch += fieldElements;
        stageYMin_.SetGlobalBuffer(taskScratch, tiling_->boxes);
        taskScratch += fieldElements;
        stageXMin_.SetGlobalBuffer(taskScratch, tiling_->boxes);
        taskScratch += fieldElements;
        stageYMax_.SetGlobalBuffer(taskScratch, tiling_->boxes);
        taskScratch += fieldElements;
        stageXMax_.SetGlobalBuffer(taskScratch, tiling_->boxes);
        taskScratch += fieldElements;
        stageArea_.SetGlobalBuffer(taskScratch, tiling_->boxes);
    }

    __aicore__ void CopyIn(LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t offset, int64_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], copyParams, padParams);
    }

    __aicore__ void CopyOut(GlobalTensor<float>& dst, int64_t offset, LocalTensor<float>& src, int64_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ void LoadScores(int64_t offset, int64_t count)
    {
        CopyIn(scoreLocal_, stageScores_, offset, count);
        const event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    }

    __aicore__ void LoadTile(int64_t offset, int64_t count)
    {
        CopyIn(scoreLocal_, stageScores_, offset, count);
        CopyIn(yMinLocal_, stageYMin_, offset, count);
        CopyIn(xMinLocal_, stageXMin_, offset, count);
        CopyIn(yMaxLocal_, stageYMax_, offset, count);
        CopyIn(xMaxLocal_, stageXMax_, offset, count);
        CopyIn(areaLocal_, stageArea_, offset, count);
        const event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    }

    __aicore__ bool FindBestCandidate(int64_t& bestIndex, float& bestScore, float scoreThreshold)
    {
        bestIndex = -1;
        bestScore = kNegativeInfinity;
        for (int64_t offset = 0; offset < tiling_->boxes; offset += tiling_->tileSize) {
            const int64_t count = MinInt64(tiling_->boxes - offset, tiling_->tileSize);
            LoadScores(offset, count);
            ReduceMax<float>(reduceOutputLocal_, scoreLocal_, reduceWorkLocal_, static_cast<int32_t>(count), false);
            PipeBarrier<PIPE_V>();
            const event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            const float tileScore = reduceOutputLocal_.GetValue(0);
            CompareScalar(compareMaskLocal_, scoreLocal_, tileScore, CMPMODE::EQ, count);
            ArithProgression<float>(temp0Local_, 0.0F, -1.0F, static_cast<int32_t>(count));
            Select(temp0Local_, compareMaskLocal_, temp0Local_, -static_cast<float>(count),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(reduceOutputLocal_, temp0Local_, reduceWorkLocal_, static_cast<int32_t>(count), false);
            PipeBarrier<PIPE_V>();
            const event_t eventIndexVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIndexVS);
            WaitFlag<HardEvent::V_S>(eventIndexVS);
            const int64_t tileIndex = -static_cast<int64_t>(reduceOutputLocal_.GetValue(0));
            const int64_t candidateIndex = offset + tileIndex;
            if (tileScore > bestScore || (tileScore == bestScore && (bestIndex < 0 || candidateIndex < bestIndex))) {
                bestScore = tileScore;
                bestIndex = candidateIndex;
            }
        }
        return bestIndex >= 0 && bestScore > scoreThreshold;
    }

    __aicore__ void RemoveSelected(int64_t bestIndex)
    {
        Duplicate(temp0Local_, kNegativeInfinity, 1);
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        CopyOut(stageScores_, bestIndex, temp0Local_, 1);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void SuppressBySelectedBox(float selectedYMin, float selectedXMin, float selectedYMax,
                                          float selectedXMax, float selectedArea, float iouThreshold)
    {
        for (int64_t offset = 0; offset < tiling_->boxes; offset += tiling_->tileSize) {
            const int64_t count = MinInt64(tiling_->boxes - offset, tiling_->tileSize);
            LoadTile(offset, count);
            Maxs(temp0Local_, yMinLocal_, selectedYMin, count);
            Mins(temp1Local_, yMaxLocal_, selectedYMax, count);
            Sub(temp0Local_, temp1Local_, temp0Local_, count);
            Maxs(temp0Local_, temp0Local_, 0.0F, count);
            Maxs(temp1Local_, xMinLocal_, selectedXMin, count);
            Mins(temp2Local_, xMaxLocal_, selectedXMax, count);
            Sub(temp1Local_, temp2Local_, temp1Local_, count);
            Maxs(temp1Local_, temp1Local_, 0.0F, count);
            Mul(temp2Local_, temp0Local_, temp1Local_, count);
            Adds(temp3Local_, areaLocal_, selectedArea, count);
            Sub(temp3Local_, temp3Local_, temp2Local_, count);
            // Preserve every positive union exactly, including sub-1e-12
            // boxes. Only substitute the denominator for invalid unions and
            // then force their IoU to zero, matching ComputeIoU.
            CompareScalar(compareMaskLocal_, temp3Local_, 0.0F, CMPMODE::GT, count);
            Duplicate(temp0Local_, 1.0F, count);
            Select(temp3Local_, compareMaskLocal_, temp3Local_, temp0Local_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            Div(temp2Local_, temp2Local_, temp3Local_, count);
            Duplicate(temp0Local_, 0.0F, count);
            Select(temp2Local_, compareMaskLocal_, temp2Local_, temp0Local_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            CompareScalar(compareMaskLocal_, temp2Local_, iouThreshold, CMPMODE::GT, count);
            Duplicate(temp3Local_, kNegativeInfinity, count);
            Select(scoreLocal_, compareMaskLocal_, temp3Local_, scoreLocal_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_ALL>();
            CopyOut(stageScores_, offset, scoreLocal_, count);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ void StoreClassIndex(int64_t resultOffset, int64_t boxIndex)
    {
        Duplicate(intScalarLocal_, static_cast<int32_t>(boxIndex), 1);
        PipeBarrier<PIPE_ALL>();
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(classIndicesGm_[resultOffset], intScalarLocal_, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void StoreClassCount(int64_t taskIndex, int64_t selectedCount)
    {
        Duplicate(intScalarLocal_, static_cast<int32_t>(selectedCount), 1);
        PipeBarrier<PIPE_ALL>();
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(classCountsGm_[taskIndex], intScalarLocal_, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void ProcessClass(int64_t taskIndex, int64_t maxOutputPerClass, float iouThreshold, float scoreThreshold)
    {
        SetTaskWorkspace(taskIndex);
        const int64_t resultBase = taskIndex * tiling_->maxOutputPerClass;
        int64_t selectedCount = 0;
        while (selectedCount < maxOutputPerClass) {
            int64_t bestIndex = -1;
            float bestScore = kNegativeInfinity;
            if (!FindBestCandidate(bestIndex, bestScore, scoreThreshold)) {
                break;
            }
            DataCopyExtParams selectedCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> selectedPadParams{false, 0, 0, 0};
            DataCopyPad(selectedBoxLocal_, stageYMin_[bestIndex], selectedCopyParams, selectedPadParams);
            DataCopyPad(selectedBoxLocal_[kScalarSlotElements], stageXMin_[bestIndex], selectedCopyParams,
                        selectedPadParams);
            DataCopyPad(selectedBoxLocal_[2 * kScalarSlotElements], stageYMax_[bestIndex], selectedCopyParams,
                        selectedPadParams);
            DataCopyPad(selectedBoxLocal_[3 * kScalarSlotElements], stageXMax_[bestIndex], selectedCopyParams,
                        selectedPadParams);
            DataCopyPad(selectedBoxLocal_[4 * kScalarSlotElements], stageArea_[bestIndex], selectedCopyParams,
                        selectedPadParams);
            const event_t eventMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMte2S);
            WaitFlag<HardEvent::MTE2_S>(eventMte2S);
            const float selectedYMin = selectedBoxLocal_.GetValue(0);
            const float selectedXMin = selectedBoxLocal_.GetValue(kScalarSlotElements);
            const float selectedYMax = selectedBoxLocal_.GetValue(2 * kScalarSlotElements);
            const float selectedXMax = selectedBoxLocal_.GetValue(3 * kScalarSlotElements);
            const float selectedArea = selectedBoxLocal_.GetValue(4 * kScalarSlotElements);
            StoreClassIndex(resultBase + selectedCount, bestIndex);
            RemoveSelected(bestIndex);
            SuppressBySelectedBox(selectedYMin, selectedXMin, selectedYMax, selectedXMax, selectedArea, iouThreshold);
            ++selectedCount;
        }
        StoreClassCount(taskIndex, selectedCount);
    }

    __aicore__ void ReadIndex(int64_t batchIndex, int64_t classIndex, int64_t boxIndex, int32_t& outputBatch,
                              int32_t& outputClass, int32_t& outputIndex) const
    {
        if (tiling_->hasIndex == 0) {
            outputBatch = static_cast<int32_t>(batchIndex);
            outputClass = static_cast<int32_t>(classIndex);
            outputIndex = static_cast<int32_t>(boxIndex);
            return;
        }
        const int64_t indexWidth = tiling_->indexWidth;
        const int64_t indexOffset = ((batchIndex * tiling_->classes + classIndex) * tiling_->boxes + boxIndex) *
                                    indexWidth;
        outputBatch = ConvertIndexValue(static_cast<float>(index_[indexOffset]));
        outputClass = ConvertIndexValue(static_cast<float>(index_[indexOffset + 1]));
        if (indexWidth == 3) {
            outputIndex = ConvertIndexValue(static_cast<float>(index_[indexOffset + 2]));
            return;
        }
        const int32_t high = ConvertIndexValue(static_cast<float>(index_[indexOffset + 2]));
        const int32_t low = ConvertIndexValue(static_cast<float>(index_[indexOffset + 3]));
        // PyTorch performs this expression in int32 and wraps modulo 2^32.
        const uint32_t merged = static_cast<uint32_t>(high) * 1000U + static_cast<uint32_t>(low);
        outputIndex = static_cast<int32_t>(merged);
    }

    __aicore__ void WriteOutputSmall(int64_t batchIndex, int64_t classIndex, int64_t boxIndex, int64_t outputOffset)
    {
        int32_t outputBatch = static_cast<int32_t>(batchIndex);
        int32_t outputClass = static_cast<int32_t>(classIndex);
        int32_t outputIndex = static_cast<int32_t>(boxIndex);
        ReadIndex(batchIndex, classIndex, boxIndex, outputBatch, outputClass, outputIndex);
        out_[outputOffset * 3] = outputBatch;
        out_[outputOffset * 3 + 1] = outputClass;
        out_[outputOffset * 3 + 2] = outputIndex;
    }

    __aicore__ void BufferOutputRow(int64_t batchIndex, int64_t classIndex, int64_t boxIndex, int64_t localRow)
    {
        int32_t outputBatch = static_cast<int32_t>(batchIndex);
        int32_t outputClass = static_cast<int32_t>(classIndex);
        int32_t outputIndex = static_cast<int32_t>(boxIndex);
        ReadIndex(batchIndex, classIndex, boxIndex, outputBatch, outputClass, outputIndex);
        SetOutputRow(localRow, outputBatch, outputClass, outputIndex);
    }

    __aicore__ void SetOutputRow(int64_t localRow, int32_t batchIndex, int32_t classIndex, int32_t boxIndex)
    {
        const int64_t offset = localRow * 3;
        intScalarLocal_.SetValue(offset, batchIndex);
        intScalarLocal_.SetValue(offset + 1, classIndex);
        intScalarLocal_.SetValue(offset + 2, boxIndex);
    }

    __aicore__ void FlushOutputRows(int64_t globalRow, int64_t rowCount)
    {
        PipeBarrier<PIPE_ALL>();
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rowCount * 3 * static_cast<int64_t>(sizeof(int32_t))), 0,
                                     0, 0};
        DataCopyPad(outGm_[globalRow * 3], intScalarLocal_, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void MergeOutput(int64_t taskCount)
    {
        int64_t written = 0;
        int64_t bufferedRows = 0;
        int64_t chunkStart = 0;
        for (int64_t taskIndex = 0; taskIndex < taskCount && written < tiling_->maxOutputSize; ++taskIndex) {
            const int64_t batchIndex = taskIndex / tiling_->classes;
            const int64_t classIndex = taskIndex % tiling_->classes;
            int64_t count = static_cast<int64_t>(classCountsGm_.GetValue(taskIndex));
            if (count < 0) {
                count = 0;
            } else if (count > tiling_->maxOutputPerClass) {
                count = tiling_->maxOutputPerClass;
            }
            const int64_t resultBase = taskIndex * tiling_->maxOutputPerClass;
            for (int64_t resultIndex = 0; resultIndex < count && written < tiling_->maxOutputSize; ++resultIndex) {
                const int64_t boxIndex = static_cast<int64_t>(classIndicesGm_.GetValue(resultBase + resultIndex));
                if (boxIndex < 0 || boxIndex >= tiling_->boxes) {
                    continue;
                }
                BufferOutputRow(batchIndex, classIndex, boxIndex, bufferedRows);
                ++bufferedRows;
                ++written;
                if (bufferedRows == kOutputTileRows) {
                    FlushOutputRows(chunkStart, bufferedRows);
                    bufferedRows = 0;
                    chunkStart = written;
                }
            }
        }
        while (written < tiling_->maxOutputSize) {
            SetOutputRow(bufferedRows, -1, -1, -1);
            ++bufferedRows;
            ++written;
            if (bufferedRows == kOutputTileRows) {
                FlushOutputRows(chunkStart, bufferedRows);
                bufferedRows = 0;
                chunkStart = written;
            }
        }
        if (bufferedRows > 0) {
            FlushOutputRows(chunkStart, bufferedRows);
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> scoreBuffer_;
    TBuf<QuePosition::VECCALC> yMinBuffer_;
    TBuf<QuePosition::VECCALC> xMinBuffer_;
    TBuf<QuePosition::VECCALC> yMaxBuffer_;
    TBuf<QuePosition::VECCALC> xMaxBuffer_;
    TBuf<QuePosition::VECCALC> areaBuffer_;
    // Shared vector scratch buffers. Candidate selection uses temp0 for the
    // tie-breaking index. IoU suppression uses temp0/temp1 for intersection
    // dimensions, temp2 for intersection/IoU, and temp3 for union/suppression.
    TBuf<QuePosition::VECCALC> temp0Buffer_;
    TBuf<QuePosition::VECCALC> temp1Buffer_;
    TBuf<QuePosition::VECCALC> temp2Buffer_;
    TBuf<QuePosition::VECCALC> temp3Buffer_;
    TBuf<QuePosition::VECCALC> reduceWorkBuffer_;
    TBuf<QuePosition::VECCALC> reduceOutputBuffer_;
    TBuf<QuePosition::VECCALC> compareMaskBuffer_;
    TBuf<QuePosition::VECCALC> intScalarBuffer_;
    TBuf<QuePosition::VECCALC> selectedBoxBuffer_;
    LocalTensor<float> scoreLocal_;
    LocalTensor<float> yMinLocal_;
    LocalTensor<float> xMinLocal_;
    LocalTensor<float> yMaxLocal_;
    LocalTensor<float> xMaxLocal_;
    LocalTensor<float> areaLocal_;
    // Local views of the shared scratch buffers described above.
    LocalTensor<float> temp0Local_;
    LocalTensor<float> temp1Local_;
    LocalTensor<float> temp2Local_;
    LocalTensor<float> temp3Local_;
    LocalTensor<float> reduceWorkLocal_;
    LocalTensor<float> reduceOutputLocal_;
    LocalTensor<uint8_t> compareMaskLocal_;
    LocalTensor<int32_t> intScalarLocal_;
    LocalTensor<float> selectedBoxLocal_;
    GlobalTensor<float> stageScores_;
    GlobalTensor<float> stageYMin_;
    GlobalTensor<float> stageXMin_;
    GlobalTensor<float> stageYMax_;
    GlobalTensor<float> stageXMax_;
    GlobalTensor<float> stageArea_;
    GlobalTensor<int32_t> classIndicesGm_;
    GlobalTensor<int32_t> classCountsGm_;
    GlobalTensor<int32_t> outGm_;
    __gm__ TBoxes* boxes_{nullptr};
    __gm__ TScores* scores_{nullptr};
    __gm__ int32_t* maxOutput_{nullptr};
    __gm__ float* iou_{nullptr};
    __gm__ float* score_{nullptr};
    __gm__ half* index_{nullptr};
    __gm__ int32_t* out_{nullptr};
    __gm__ int32_t* selectedBoxes_{nullptr};
    GM_ADDR userWorkspace_{nullptr};
    __gm__ float* scratch_{nullptr};
    const NonMaxSuppressionV7TilingData* tiling_{nullptr};
    bool workspaceReady_{false};
};
} // namespace NonMaxSuppressionV7Op

#endif
