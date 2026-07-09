/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_align_grad.h
 * \brief RoiAlignGrad kernel implementation.
 */

#ifndef __ROI_ALIGN_GRAD_H__
#define __ROI_ALIGN_GRAD_H__

#include <cstdint>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "roi_align_grad_tiling_data.h"
#include "roi_align_grad_tiling_key.h"

namespace NsRoiAlignGrad {

using namespace AscendC;

constexpr uint64_t kC0Size = 16U;
constexpr uint32_t kRoiElemNum = 5U;
constexpr uint32_t kRoiBufferBytes = 32U;
constexpr uint32_t kSyncIntNumPerCore = 8U * 32U;
constexpr uint64_t kInitGlobalMemoryMaxCount = 65528U;
constexpr uint64_t kNdZeroBufferElemNum = 1024U;
// yIndex 归并上限：一个 poolH 内 sampleH 个 grid 点最多产生 2*sampleH 个 (low/high) 命中，
// 落在 xDiffH 上的不同整数行数远小于此。128 足够覆盖极大 sampling_ratio；超界回退逐点。
constexpr int32_t kMaxYHits = 128;

struct AxisPoint {
    int32_t low = 0;
    int32_t high = 0;
    float lowWeight = 0.0F;
    float highWeight = 0.0F;
};

struct RoiBox {
    float batchIdx = 0.0F;
    float startX = 0.0F;
    float startY = 0.0F;
    float endX = 0.0F;
    float endY = 0.0F;
};

template <typename T>
class RoiAlignGrad {
public:
    __aicore__ inline RoiAlignGrad() {}

    __aicore__ inline void Init(GM_ADDR yDiff, GM_ADDR rois, GM_ADDR xDiff, GM_ADDR workspace,
                                const RoiAlignGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const RoiAlignGradTilingData* tilingData);
    __aicore__ inline void InitGlobalMemory(GM_ADDR yDiff, GM_ADDR rois, GM_ADDR xDiff, GM_ADDR workspace);
    __aicore__ inline void InitCoreSchedule();
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGlobalZero(uint64_t start, uint64_t count);
    __aicore__ inline void ZeroOutput();
    __aicore__ inline void ZeroSyncWorkspace();
    __aicore__ inline void SyncAfterZero();
    __aicore__ inline void LoadRoi(uint64_t roiIndex, RoiBox& roi);
    __aicore__ inline void ProcessOneRoi(uint64_t roiIndex, uint64_t c1Start, uint64_t c1Num);
    __aicore__ inline void ProcessOneRoiMoveOneRowSumInUb(uint64_t roiIndex, uint64_t c1Start, uint64_t c1Num);
    __aicore__ inline void AccumulateMoveOneRowOutputRow(const LocalTensor<T>& yDiffRowLocal, uint64_t xDiffBase,
                                                         int32_t yIndex, float yWeight, int32_t xSampleCount,
                                                         int32_t activeXMin, int32_t activeXMax);
    __aicore__ inline void AccumulateNdOutputRow(uint64_t roiIndex, uint64_t cIdx, int32_t poolH,
                                                 uint64_t xDiffRowOffset, float yWeight, float roiStartX,
                                                 float binSizeW, float gridW, int32_t sampleW);
    __aicore__ inline void ZeroNdOutputChannel(uint64_t xDiffBase);
    __aicore__ inline void ProcessNdOutputChannel(uint64_t nIdx, uint64_t cIdx);
    __aicore__ inline void ProcessC1PlaneNc1hwc0(uint64_t c1Idx);
    __aicore__ inline void ProcessOneRoiNc1hwc0Scalar(uint64_t roiIndex, uint64_t cStart, uint64_t cNum);
    __aicore__ inline void ProcessOneRoiNd(uint64_t roiIndex, uint64_t cStart, uint64_t cNum);
    __aicore__ inline void CalcC1Range(uint64_t coreRoiIndex, uint64_t& c1Start, uint64_t& c1Num) const;
    __aicore__ inline AxisPoint CalcAxisPoint(float coord, int32_t limit, int32_t samples) const;
    __aicore__ inline float CalcGridCoordinate(float startCoordinate, float gridDistance, int32_t sampleCount,
                                               int32_t poolIndex, int32_t gridIndex) const;
    __aicore__ inline bool NoSampleOverlap(float roiStartX, float roiStartY, float gridW, float gridH, int32_t sampleW,
                                           int32_t sampleH) const;
    __aicore__ inline void AtomicAddVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset, float weight);
    __aicore__ inline void AtomicAddVectorActive(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset, float weight);
    __aicore__ inline void AccumulateGlobalVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset, float weight);
    __aicore__ inline void AtomicAddScalar(T value, uint64_t xDiffOffset);
    __aicore__ inline void AccumulateLocalVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset, float weight);
    __aicore__ inline static void MergeYHit(int32_t* hitIdx, float* hitWeight, int32_t& hitCount, int32_t index,
                                            float weight, bool& overflow);

    __aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) const;
    __aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs) const;
    __aicore__ inline uint64_t MaxU64(uint64_t lhs, uint64_t rhs) const;
    __aicore__ inline float MinF32(float lhs, float rhs) const;
    __aicore__ inline float MaxF32(float lhs, float rhs) const;
    __aicore__ inline int32_t FloorToInt(float value) const;
    __aicore__ inline int32_t CeilToInt(float value) const;

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> roiBuf_;
    TBuf<TPosition::VECIN> yDiffBuf_;
    TBuf<TPosition::VECOUT> xDiffBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> syncBuf_;
    // 第3级优化：X 采样点预计算复用（CSE）。存有效采样点的 low/high(int32) 和
    // lowWeight/highWeight(float)，供 poolH×sampleH×角 循环复用，避免重复标量坐标计算。
    TBuf<TPosition::VECCALC> xIdxBuf_;    // int32: [low0,high0,low1,high1,...]
    TBuf<TPosition::VECCALC> xWeightBuf_; // float: [lw0,hw0,lw1,hw1,...]

    GlobalTensor<T> yDiffGm_;
    GlobalTensor<float> roisGm_;
    GlobalTensor<T> xDiffGm_;
    GlobalTensor<int32_t> syncGm_;

    uint64_t tilingKey_ = ROI_ALIGN_GRAD_TPL_SCH_DEFAULT;
    uint64_t runningCoreNum_ = 1U;
    uint64_t roiCount_ = 0U;
    uint64_t roisRowSize_ = 0U;
    uint64_t xDiffN_ = 0U;
    uint64_t xDiffC_ = 0U;
    uint64_t c1_ = 0U;
    uint64_t xDiffH_ = 0U;
    uint64_t xDiffW_ = 0U;
    uint64_t c1BatchMax_ = 1U;

    int32_t pooledWidth_ = 0;
    int32_t pooledHeight_ = 0;
    int32_t sampleNum_ = 0;
    int32_t roiEndMode_ = 0;
    int32_t isNd_ = 0;

    float spatialScale_ = 0.0F;
    float pooledWidthReciprocal_ = 0.0F;
    float pooledHeightReciprocal_ = 0.0F;
    float sampleNumReciprocal_ = 0.0F;

    uint64_t coreNc_ = 0U;
    uint64_t coreNcOffset_ = 0U;
    uint64_t coreRoiOffset_ = 0U;
    uint64_t coreRoiCount_ = 0U;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::Init(GM_ADDR yDiff, GM_ADDR rois, GM_ADDR xDiff, GM_ADDR workspace,
                                             const RoiAlignGradTilingData* tilingData)
{
    InitTiling(tilingData);
    InitGlobalMemory(yDiff, rois, xDiff, workspace);
    InitCoreSchedule();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::InitTiling(const RoiAlignGradTilingData* tilingData)
{
    tilingKey_ = tilingData->tilingKey;
    runningCoreNum_ = tilingData->runningCoreNum;
    roiCount_ = tilingData->roiCount;
    roisRowSize_ = tilingData->roisRowSize;
    xDiffN_ = tilingData->xDiffN;
    xDiffC_ = tilingData->xDiffC;
    c1_ = tilingData->c1;
    xDiffH_ = tilingData->xDiffH;
    xDiffW_ = tilingData->xDiffW;
    c1BatchMax_ = tilingData->c1BatchMax;
    pooledWidth_ = tilingData->pooledWidth;
    pooledHeight_ = tilingData->pooledHeight;
    sampleNum_ = tilingData->sampleNum;
    roiEndMode_ = tilingData->roiEndMode;
    isNd_ = tilingData->isNd;
    spatialScale_ = tilingData->spatialScale;
    pooledWidthReciprocal_ = tilingData->pooledWidthReciprocal;
    pooledHeightReciprocal_ = tilingData->pooledHeightReciprocal;
    sampleNumReciprocal_ = tilingData->sampleNumReciprocal;
    blockIdx_ = GetBlockIdx();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::InitGlobalMemory(GM_ADDR yDiff, GM_ADDR rois, GM_ADDR xDiff, GM_ADDR workspace)
{
    const uint64_t channelStride = isNd_ != 0 ? 1U : kC0Size;
    const uint64_t yDiffElemNum = roiCount_ * c1_ * static_cast<uint64_t>(pooledHeight_) *
                                  static_cast<uint64_t>(pooledWidth_) * channelStride;
    const uint64_t roisElemNum = roiCount_ * roisRowSize_;
    const uint64_t xDiffElemNum = xDiffN_ * c1_ * xDiffH_ * xDiffW_ * channelStride;
    const uint64_t syncElemNum = runningCoreNum_ * static_cast<uint64_t>(kSyncIntNumPerCore);

    yDiffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(yDiff), static_cast<uint32_t>(yDiffElemNum));
    roisGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rois), static_cast<uint32_t>(roisElemNum));
    xDiffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xDiff), static_cast<uint32_t>(xDiffElemNum));
    syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), static_cast<uint32_t>(syncElemNum));
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::InitCoreSchedule()
{
    if (runningCoreNum_ == 0U || c1_ == 0U) {
        coreNc_ = 0U;
        coreNcOffset_ = 0U;
        coreRoiOffset_ = 0U;
        coreRoiCount_ = 0U;
        return;
    }

    const uint64_t scheduleN = isNd_ != 0 ? xDiffN_ : roiCount_;
    const uint64_t totalNc = scheduleN * c1_;
    const uint64_t ncPerCore = totalNc / runningCoreNum_;
    const uint64_t ncPerCoreTail = totalNc % runningCoreNum_;

    const bool useNc1hwc0RoiSplit = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;
    if (isNd_ == 0 && !useNc1hwc0RoiSplit) {
        const uint64_t c1PerCore = CeilDiv(c1_, runningCoreNum_);
        coreNcOffset_ = static_cast<uint64_t>(blockIdx_) * c1PerCore;
        coreNc_ = coreNcOffset_ < c1_ ? MinU64(c1PerCore, c1_ - coreNcOffset_) : 0U;
        coreRoiOffset_ = 0U;
        coreRoiCount_ = coreNc_ > 0U ? roiCount_ : 0U;
        return;
    }

    const uint64_t blockIndex = static_cast<uint64_t>(blockIdx_);
    if (isNd_ == 0 && useNc1hwc0RoiSplit) {
        if (blockIndex == 0U) {
            coreNc_ = ncPerCore + ncPerCoreTail;
            coreNcOffset_ = 0U;
        } else {
            coreNc_ = ncPerCore;
            coreNcOffset_ = blockIndex * ncPerCore + ncPerCoreTail;
        }
    } else if (blockIndex < ncPerCoreTail) {
        coreNc_ = ncPerCore + 1U;
        coreNcOffset_ = blockIndex * coreNc_;
    } else {
        coreNc_ = ncPerCore;
        coreNcOffset_ = ncPerCoreTail * (ncPerCore + 1U) + (blockIndex - ncPerCoreTail) * ncPerCore;
    }

    if (coreNc_ == 0U) {
        coreRoiOffset_ = 0U;
        coreRoiCount_ = 0U;
        return;
    }

    coreRoiOffset_ = coreNcOffset_ / c1_;
    const uint64_t coreRoiEnd = (coreNcOffset_ + coreNc_ - 1U) / c1_;
    coreRoiCount_ = coreRoiEnd - coreRoiOffset_ + 1U;
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::InitLocalBuffer()
{
    if (isNd_ != 0) {
        pipe_.InitBuffer(roiBuf_, kRoiBufferBytes);
        pipe_.InitBuffer(yDiffBuf_, kRoiBufferBytes);
        const uint64_t xDiffRowBytes = xDiffW_ * sizeof(T);
        const uint64_t zeroBufferBytes = kNdZeroBufferElemNum * sizeof(T);
        pipe_.InitBuffer(xDiffBuf_, static_cast<uint32_t>(CeilDiv(MaxU64(xDiffRowBytes, zeroBufferBytes), 32U) * 32U));
        pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(CeilDiv(MaxU64(xDiffRowBytes, zeroBufferBytes), 32U) * 32U));
        pipe_.InitBuffer(syncBuf_, static_cast<uint32_t>(runningCoreNum_ * static_cast<uint64_t>(kSyncIntNumPerCore) *
                                                         sizeof(int32_t)));
        return;
    }

    uint64_t yDiffBufElemNum = c1BatchMax_ * kC0Size;
    if (tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW || tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT) {
        yDiffBufElemNum *= static_cast<uint64_t>(pooledWidth_);
    }
    const bool useMoveOneRow = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW;
    const bool useNc1hwc0RoiSplit = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;
    const uint64_t xDiffBufElemNum = (useMoveOneRow || useNc1hwc0RoiSplit) ? (xDiffW_ * kC0Size) :
                                                                             (xDiffH_ * xDiffW_ * kC0Size);

    pipe_.InitBuffer(roiBuf_, kRoiBufferBytes);
    pipe_.InitBuffer(yDiffBuf_, static_cast<uint32_t>(yDiffBufElemNum * sizeof(T)));
    pipe_.InitBuffer(xDiffBuf_, static_cast<uint32_t>(xDiffBufElemNum * sizeof(T)));
    pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(kC0Size * sizeof(T)));
    // 第3级：X 采样点预计算 buffer。仅 sampleNum_>0（固定采样数）时按 pooledWidth_×sampleNum_
    // 分配；sampleNum_≤0（动态 sampleW）走 fallback，此处占位最小分配。每采样点存 2 个索引
    // (low/high) 和 2 个权重 (lw/hw)。
    const uint64_t maxXSamples = static_cast<uint64_t>(pooledWidth_) *
                                 static_cast<uint64_t>(sampleNum_ > 0 ? sampleNum_ : 1);
    // xIdxBuf 每采样点存 4 个 int32 [poolW, low, high, lowActive]；xWeightBuf 每采样点 2 个 float [lw, hw]
    pipe_.InitBuffer(xIdxBuf_, static_cast<uint32_t>(maxXSamples * 4U * sizeof(int32_t)));
    pipe_.InitBuffer(xWeightBuf_, static_cast<uint32_t>(maxXSamples * 2U * sizeof(float)));
    pipe_.InitBuffer(
        syncBuf_, static_cast<uint32_t>(runningCoreNum_ * static_cast<uint64_t>(kSyncIntNumPerCore) * sizeof(int32_t)));
}

template <typename T>
__aicore__ inline uint64_t RoiAlignGrad<T>::CeilDiv(uint64_t value, uint64_t divisor) const
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

template <typename T>
__aicore__ inline uint64_t RoiAlignGrad<T>::MinU64(uint64_t lhs, uint64_t rhs) const
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline uint64_t RoiAlignGrad<T>::MaxU64(uint64_t lhs, uint64_t rhs) const
{
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline float RoiAlignGrad<T>::MinF32(float lhs, float rhs) const
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline float RoiAlignGrad<T>::MaxF32(float lhs, float rhs) const
{
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline int32_t RoiAlignGrad<T>::FloorToInt(float value) const
{
    int32_t truncated = static_cast<int32_t>(value);
    if (static_cast<float>(truncated) > value) {
        truncated -= 1;
    }
    return truncated;
}

template <typename T>
__aicore__ inline int32_t RoiAlignGrad<T>::CeilToInt(float value) const
{
    int32_t truncated = static_cast<int32_t>(value);
    if (static_cast<float>(truncated) < value) {
        truncated += 1;
    }
    return truncated;
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::InitGlobalZero(uint64_t start, uint64_t count)
{
    for (uint64_t offset = 0U; offset < count; offset += kInitGlobalMemoryMaxCount) {
        const uint64_t curCount = MinU64(kInitGlobalMemoryMaxCount, count - offset);
        GlobalTensor<T> xDiffInitGm;
        xDiffInitGm.SetGlobalBuffer(xDiffGm_.GetPhyAddr(start + offset), static_cast<uint32_t>(curCount));
        AscendC::InitGlobalMemory<T>(xDiffInitGm, static_cast<uint32_t>(curCount), static_cast<T>(0));
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ZeroOutput()
{
    if (runningCoreNum_ == 0U) {
        return;
    }

    const uint64_t channelStride = isNd_ != 0 ? 1U : kC0Size;
    if (isNd_ != 0) {
        const uint64_t channelElemNum = xDiffH_ * xDiffW_ * channelStride;
        const uint64_t outputChannelNum = xDiffN_ * c1_;
        const uint64_t channelNumPerCore = CeilDiv(outputChannelNum, runningCoreNum_);
        const uint64_t channelStart = static_cast<uint64_t>(blockIdx_) * channelNumPerCore;
        if (channelStart >= outputChannelNum) {
            return;
        }

        const uint64_t channelNum = MinU64(channelNumPerCore, outputChannelNum - channelStart);
        for (uint64_t channelOffset = 0U; channelOffset < channelNum; ++channelOffset) {
            InitGlobalZero((channelStart + channelOffset) * channelElemNum, channelElemNum);
        }
        return;
    }

    const bool useNc1hwc0RoiSplit = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;
    const bool useMoveOneRow = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW;
    if (isNd_ == 0 && useNc1hwc0RoiSplit) {
        const uint64_t planeElemNum = xDiffH_ * xDiffW_ * kC0Size;
        const uint64_t outputPlaneNum = xDiffN_ * c1_;
        const uint64_t planeNumPerCore = CeilDiv(outputPlaneNum, runningCoreNum_);
        const uint64_t planeStart = static_cast<uint64_t>(blockIdx_) * planeNumPerCore;
        if (planeStart >= outputPlaneNum) {
            return;
        }

        const uint64_t planeNum = MinU64(planeNumPerCore, outputPlaneNum - planeStart);
        for (uint64_t planeOffset = 0U; planeOffset < planeNum; ++planeOffset) {
            InitGlobalZero((planeStart + planeOffset) * planeElemNum, planeElemNum);
        }
        return;
    }

    if (isNd_ == 0 && useMoveOneRow) {
        if (coreNc_ == 0U) {
            return;
        }

        const uint64_t planeElemNum = xDiffH_ * xDiffW_ * kC0Size;
        for (uint64_t nIdx = 0U; nIdx < xDiffN_; ++nIdx) {
            for (uint64_t c1Offset = 0U; c1Offset < coreNc_; ++c1Offset) {
                const uint64_t c1Idx = coreNcOffset_ + c1Offset;
                const uint64_t planeBase = (nIdx * c1_ + c1Idx) * planeElemNum;
                InitGlobalZero(planeBase, planeElemNum);
            }
        }
        return;
    }

    if (isNd_ == 0 && !useNc1hwc0RoiSplit) {
        if (coreNc_ == 0U) {
            return;
        }

        const uint64_t planeElemNum = xDiffH_ * xDiffW_ * kC0Size;
        for (uint64_t nIdx = 0U; nIdx < xDiffN_; ++nIdx) {
            for (uint64_t c1Offset = 0U; c1Offset < coreNc_; ++c1Offset) {
                const uint64_t c1Idx = coreNcOffset_ + c1Offset;
                const uint64_t start = (nIdx * c1_ + c1Idx) * planeElemNum;
                InitGlobalZero(start, planeElemNum);
            }
        }
        return;
    }

    const uint64_t totalElemNum = xDiffN_ * c1_ * xDiffH_ * xDiffW_ * channelStride;
    const uint64_t elemNumPerCore = CeilDiv(totalElemNum, runningCoreNum_);
    const uint64_t start = static_cast<uint64_t>(blockIdx_) * elemNumPerCore;
    if (start >= totalElemNum) {
        return;
    }

    const uint64_t count = MinU64(elemNumPerCore, totalElemNum - start);
    InitGlobalZero(start, count);
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ZeroSyncWorkspace()
{
    if (runningCoreNum_ <= 1U) {
        return;
    }

    const uint64_t start = static_cast<uint64_t>(blockIdx_) * static_cast<uint64_t>(kSyncIntNumPerCore);
    GlobalTensor<int32_t> syncInitGm;
    syncInitGm.SetGlobalBuffer(syncGm_.GetPhyAddr(start), static_cast<uint32_t>(kSyncIntNumPerCore));
    AscendC::InitGlobalMemory<int32_t>(syncInitGm, static_cast<uint32_t>(kSyncIntNumPerCore), 0);
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::SyncAfterZero()
{
    if (runningCoreNum_ <= 1U) {
        return;
    }
    // 用无参硬件 SyncAll（ffts cross-core sync，纯硬件 barrier，不访问 GM/UB workspace），
    // 避免三参数软件 SyncAll 的 workspace 布局/越界问题（Test_025 等 blockDim<40 的 RoiSplit case
    // 会触发 MTE DDR 地址越界崩溃）。AIVOnly 场景。
    AscendC::SyncAll<true>();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::LoadRoi(uint64_t roiIndex, RoiBox& roi)
{
    const uint64_t roiOffset = roiIndex * roisRowSize_;
    roi.batchIdx = roisGm_.GetValue(roiOffset);
    roi.startX = roisGm_.GetValue(roiOffset + 1U);
    roi.startY = roisGm_.GetValue(roiOffset + 2U);
    roi.endX = roisGm_.GetValue(roiOffset + 3U);
    roi.endY = roisGm_.GetValue(roiOffset + 4U);
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::CalcC1Range(uint64_t coreRoiIndex, uint64_t& c1Start, uint64_t& c1Num) const
{
    const uint64_t roiIndex = coreRoiOffset_ + coreRoiIndex;
    const uint64_t coreNcBegin = coreNcOffset_;
    const uint64_t coreNcEnd = coreNcOffset_ + coreNc_;
    const uint64_t roiNcBegin = roiIndex * c1_;
    const uint64_t roiNcEnd = roiNcBegin + c1_;
    const uint64_t ncBegin = coreNcBegin > roiNcBegin ? coreNcBegin : roiNcBegin;
    const uint64_t ncEnd = coreNcEnd < roiNcEnd ? coreNcEnd : roiNcEnd;
    if (ncBegin >= ncEnd) {
        c1Start = 0U;
        c1Num = 0U;
        return;
    }

    c1Start = ncBegin - roiNcBegin;
    c1Num = ncEnd - ncBegin;
}

template <typename T>
__aicore__ inline AxisPoint RoiAlignGrad<T>::CalcAxisPoint(float coord, int32_t limit, int32_t samples) const
{
    AxisPoint point;
    if (limit <= 0 || samples <= 0) {
        return point;
    }

    if (coord < -1.0F || coord > static_cast<float>(limit)) {
        return point;
    }

    const float clampedCoord = MinF32(MaxF32(coord, 0.0F), static_cast<float>(limit - 1));
    const int32_t low = FloorToInt(clampedCoord);
    const int32_t high = (low + 1) < limit ? (low + 1) : (limit - 1);
    const float frac = clampedCoord - static_cast<float>(low);

    point.low = low;
    point.high = high;
    if (sampleNum_ > 0) {
        volatile float oneMinusFrac = 1.0F - frac;
        volatile float lowWeight = oneMinusFrac * sampleNumReciprocal_;
        volatile float highWeight = frac * sampleNumReciprocal_;
        point.lowWeight = lowWeight;
        point.highWeight = highWeight;
    } else {
        volatile float oneMinusFrac = 1.0F - frac;
        volatile float sampleCount = static_cast<float>(samples);
        volatile float lowWeight = oneMinusFrac / sampleCount;
        volatile float highWeight = frac / sampleCount;
        point.lowWeight = lowWeight;
        point.highWeight = highWeight;
    }
    return point;
}

template <typename T>
__aicore__ inline float RoiAlignGrad<T>::CalcGridCoordinate(float startCoordinate, float gridDistance,
                                                            int32_t sampleCount, int32_t poolIndex,
                                                            int32_t gridIndex) const
{
    const int32_t sampleIndex = poolIndex * sampleCount + gridIndex;
    const float sampleOffset = static_cast<float>(sampleIndex) + 0.5F;
    volatile float gridOffset = sampleOffset * gridDistance;
    return startCoordinate + gridOffset;
}

template <typename T>
__aicore__ inline bool RoiAlignGrad<T>::NoSampleOverlap(float roiStartX, float roiStartY, float gridW, float gridH,
                                                        int32_t sampleW, int32_t sampleH) const
{
    const float firstX = CalcGridCoordinate(roiStartX, gridW, sampleW, 0, 0);
    const float lastX = CalcGridCoordinate(roiStartX, gridW, sampleW, pooledWidth_ - 1, sampleW - 1);
    const float minX = MinF32(firstX, lastX);
    const float maxX = MaxF32(firstX, lastX);
    const int32_t xDiffWLimit = static_cast<int32_t>(xDiffW_);
    if (maxX < -1.0F || minX > static_cast<float>(xDiffWLimit)) {
        return true;
    }

    const float firstY = CalcGridCoordinate(roiStartY, gridH, sampleH, 0, 0);
    const float lastY = CalcGridCoordinate(roiStartY, gridH, sampleH, pooledHeight_ - 1, sampleH - 1);
    const float minY = MinF32(firstY, lastY);
    const float maxY = MaxF32(firstY, lastY);
    const int32_t xDiffHLimit = static_cast<int32_t>(xDiffH_);
    return maxY < -1.0F || minY > static_cast<float>(xDiffHLimit);
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AtomicAddVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset,
                                                        float weight)
{
    if (weight == 0.0F) {
        return;
    }

    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    Muls(xDiffLocal, yDiffLocal, static_cast<T>(weight), static_cast<int32_t>(kC0Size));
    PipeBarrier<PIPE_V>();
    SetAtomicAdd<T>();
    DataCopy(xDiffGm_[xDiffOffset], xDiffLocal, static_cast<uint32_t>(kC0Size));
    PipeBarrier<PIPE_MTE3>();
    SetAtomicNone();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AtomicAddVectorActive(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset,
                                                              float weight)
{
    if (weight == 0.0F) {
        return;
    }

    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    Muls(xDiffLocal, yDiffLocal, static_cast<T>(weight), static_cast<int32_t>(kC0Size));
    PipeBarrier<PIPE_V>();
    DataCopy(xDiffGm_[xDiffOffset], xDiffLocal, static_cast<uint32_t>(kC0Size));
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AccumulateGlobalVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset,
                                                               float weight)
{
    if (weight == 0.0F) {
        return;
    }

    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    LocalTensor<T> tmpLocal = tmpBuf_.template Get<T>();
    DataCopy(xDiffLocal, xDiffGm_[xDiffOffset], static_cast<uint32_t>(kC0Size));
    PipeBarrier<PIPE_ALL>();
    Muls(tmpLocal, yDiffLocal, static_cast<T>(weight), static_cast<int32_t>(kC0Size));
    PipeBarrier<PIPE_V>();
    Add(xDiffLocal, xDiffLocal, tmpLocal, static_cast<int32_t>(kC0Size));
    PipeBarrier<PIPE_V>();
    DataCopy(xDiffGm_[xDiffOffset], xDiffLocal, static_cast<uint32_t>(kC0Size));
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AtomicAddScalar(T value, uint64_t xDiffOffset)
{
    if (static_cast<float>(value) == 0.0F) {
        return;
    }

    const float oldValue = static_cast<float>(xDiffGm_.GetValue(xDiffOffset));
    xDiffGm_.SetValue(xDiffOffset, static_cast<T>(oldValue + static_cast<float>(value)));
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AccumulateLocalVector(const LocalTensor<T>& yDiffLocal, uint64_t xDiffOffset,
                                                              float weight)
{
    if (weight == 0.0F) {
        return;
    }

    // Axpy 融合乘加 dst += src × scalar，等价于原 Muls(tmp)+Add(x,x,tmp)。
    // 第2级优化：去掉内部 PIPE_V barrier。所有 Axpy 都在 PIPE_V(vector)队列内顺序
    // 执行（同队列 FIFO），同地址累加天然 RAW 保序，不同地址无依赖；与后续写回 GM 的
    // 同步由调用方末尾的 PipeBarrier<PIPE_ALL> 统一保证。省去内层循环数千次冗余 barrier。
    LocalTensor<T> xDiffLocal = xDiffBuf_.template GetWithOffset<T>(static_cast<uint32_t>(kC0Size),
                                                                    static_cast<uint32_t>(xDiffOffset * sizeof(T)));
    Axpy(xDiffLocal, yDiffLocal, static_cast<T>(weight), static_cast<int32_t>(kC0Size));
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::MergeYHit(int32_t* hitIdx, float* hitWeight, int32_t& hitCount, int32_t index,
                                                  float weight, bool& overflow)
{
    for (int32_t k = 0; k < hitCount; ++k) {
        if (hitIdx[k] == index) {
            hitWeight[k] += weight;
            return;
        }
    }
    if (hitCount < kMaxYHits) {
        hitIdx[hitCount] = index;
        hitWeight[hitCount] = weight;
        ++hitCount;
    } else {
        overflow = true;
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AccumulateMoveOneRowOutputRow(const LocalTensor<T>& yDiffRowLocal,
                                                                      uint64_t xDiffBase, int32_t yIndex, float yWeight,
                                                                      int32_t xSampleCount, int32_t activeXMin,
                                                                      int32_t activeXMax)
{
    if (yWeight == 0.0F || activeXMax < activeXMin) {
        return;
    }

    const uint64_t xDiffRowOffset = xDiffBase + static_cast<uint64_t>(yIndex) * xDiffW_ * kC0Size;
    const uint64_t activeXOffset = static_cast<uint64_t>(activeXMin) * kC0Size;
    const uint32_t activeXElemNum = static_cast<uint32_t>(activeXMax - activeXMin + 1) * static_cast<uint32_t>(kC0Size);
    LocalTensor<T> xDiffRowLocal = xDiffBuf_.template Get<T>();
    Duplicate(xDiffRowLocal[activeXOffset], static_cast<T>(0), static_cast<int32_t>(activeXElemNum));
    PipeBarrier<PIPE_V>();

    // 第3级 CSE：从预计算 buffer 读 X 采样点，省掉 CalcGridCoordinate/CalcAxisPoint 重复计算。
    // xIdxLocal 每采样点 4 元组 [poolW, low, high, lowActive]；xWeightLocal 每采样点 2 元组 [lw, hw]。
    LocalTensor<int32_t> xIdxLocal = xIdxBuf_.template Get<int32_t>();
    LocalTensor<float> xWeightLocal = xWeightBuf_.template Get<float>();
    // 第4级(2A软件流水)：预取下一采样点元数据，使标量 GetValue(下一轮) 与向量 Axpy(本轮)
    // 重叠执行，打破 GetValue->Axpy 交替串行依赖链。累加顺序不变，精度零风险。
    if (xSampleCount > 0) {
        // 第5级（批量散射）：每条条目 = 一个 (poolW,target)，权重已在预计算阶段合并。
        // 保留软件流水预取：标量 GetValue(下一条) 与向量 Axpy(本条) 重叠。
        int32_t poolW = xIdxLocal.GetValue(0);
        int32_t target = xIdxLocal.GetValue(1);
        float w = xWeightLocal.GetValue(0);
        for (int32_t s = 0; s < xSampleCount; ++s) {
            const int32_t curPoolW = poolW;
            const int32_t curTarget = target;
            const float curW = w;
            if (s + 1 < xSampleCount) {
                const int32_t n = s + 1;
                poolW = xIdxLocal.GetValue(n * 2);
                target = xIdxLocal.GetValue(n * 2 + 1);
                w = xWeightLocal.GetValue(n);
            }
            LocalTensor<T> yDiffLocal = yDiffRowLocal[static_cast<uint64_t>(curPoolW) * kC0Size];
            AccumulateLocalVector(yDiffLocal, static_cast<uint64_t>(curTarget) * kC0Size, yWeight * curW);
        }
    }

    PipeBarrier<PIPE_ALL>();
    DataCopy(xDiffGm_[xDiffRowOffset + activeXOffset], xDiffRowLocal[activeXOffset], activeXElemNum);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::AccumulateNdOutputRow(uint64_t roiIndex, uint64_t cIdx, int32_t poolH,
                                                              uint64_t xDiffRowOffset, float yWeight, float roiStartX,
                                                              float binSizeW, float gridW, int32_t sampleW)
{
    if (yWeight == 0.0F || xDiffW_ == 0U) {
        return;
    }

    const uint64_t sampleWCount = static_cast<uint64_t>(sampleW);
    const uint64_t totalSampleW = static_cast<uint64_t>(pooledWidth_) * sampleWCount;
    const bool useDirectGlobalAccumulate = roiEndMode_ == 0 && c1_ < 80U;
    if (useDirectGlobalAccumulate) {
        for (uint64_t sampleIndex = 0U; sampleIndex < totalSampleW; ++sampleIndex) {
            const int32_t poolW = static_cast<int32_t>(sampleIndex / sampleWCount);
            const int32_t gridWIdx = static_cast<int32_t>(sampleIndex - static_cast<uint64_t>(poolW) * sampleWCount);
            const uint64_t yDiffOffset = ((roiIndex * c1_ + cIdx) * static_cast<uint64_t>(pooledHeight_) *
                                              static_cast<uint64_t>(pooledWidth_) +
                                          static_cast<uint64_t>(poolH) * static_cast<uint64_t>(pooledWidth_) +
                                          static_cast<uint64_t>(poolW));
            const float yDiffValue = static_cast<float>(yDiffGm_.GetValue(yDiffOffset));
            if (yDiffValue == 0.0F) {
                continue;
            }

            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
            if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                continue;
            }

            float lowWeight = xPoint.lowWeight * yWeight;
            if (xPoint.low == xPoint.high) {
                lowWeight += xPoint.highWeight * yWeight;
            }
            const float lowValue = yDiffValue * lowWeight;
            if (lowValue != 0.0F) {
                AtomicAddScalar(static_cast<T>(lowValue), xDiffRowOffset + static_cast<uint64_t>(xPoint.low));
            }

            if (xPoint.low != xPoint.high) {
                const float highValue = yDiffValue * xPoint.highWeight * yWeight;
                if (highValue != 0.0F) {
                    AtomicAddScalar(static_cast<T>(highValue), xDiffRowOffset + static_cast<uint64_t>(xPoint.high));
                }
            }
        }
        return;
    }

    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    Duplicate(xDiffLocal, static_cast<T>(0), static_cast<int32_t>(xDiffW_));
    const bool useLocalCompensation = false;
    LocalTensor<T> compLocal = tmpBuf_.template Get<T>();
    if (useLocalCompensation) {
        Duplicate(compLocal, static_cast<T>(0), static_cast<int32_t>(xDiffW_));
    }
    PipeBarrier<PIPE_ALL>();

    const bool useTbeLikeSampleBlock = roiEndMode_ == 2 || c1_ >= 80U;
    const uint64_t accumSampleBlock = useTbeLikeSampleBlock ? MinU64(totalSampleW, 128U) : totalSampleW;
    for (uint64_t sampleBlockStart = 0U; sampleBlockStart < totalSampleW; sampleBlockStart += accumSampleBlock) {
        uint64_t minTouched = xDiffW_;
        uint64_t maxTouched = 0U;
        const uint64_t sampleBlockEnd = MinU64(sampleBlockStart + accumSampleBlock, totalSampleW);
        for (uint64_t sampleIndex = sampleBlockStart; sampleIndex < sampleBlockEnd; ++sampleIndex) {
            const int32_t poolW = static_cast<int32_t>(sampleIndex / sampleWCount);
            const int32_t gridWIdx = static_cast<int32_t>(sampleIndex - static_cast<uint64_t>(poolW) * sampleWCount);
            const uint64_t yDiffOffset = ((roiIndex * c1_ + cIdx) * static_cast<uint64_t>(pooledHeight_) *
                                              static_cast<uint64_t>(pooledWidth_) +
                                          static_cast<uint64_t>(poolH) * static_cast<uint64_t>(pooledWidth_) +
                                          static_cast<uint64_t>(poolW));
            const float yDiffValue = static_cast<float>(yDiffGm_.GetValue(yDiffOffset));
            if (yDiffValue == 0.0F) {
                continue;
            }

            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
            if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                continue;
            }

            float lowWeight = xPoint.lowWeight * yWeight;
            if (xPoint.low == xPoint.high) {
                lowWeight += xPoint.highWeight * yWeight;
            }
            const float lowValue = yDiffValue * lowWeight;
            if (lowValue != 0.0F) {
                const uint64_t lowOffset = static_cast<uint64_t>(xPoint.low);
                const float oldValue = static_cast<float>(xDiffLocal.GetValue(lowOffset));
                if (useLocalCompensation) {
                    const float compensation = static_cast<float>(compLocal.GetValue(lowOffset));
                    const float correctedValue = lowValue - compensation;
                    const float newValue = oldValue + correctedValue;
                    const float newCompensation = (newValue - oldValue) - correctedValue;
                    xDiffLocal.SetValue(lowOffset, static_cast<T>(newValue));
                    compLocal.SetValue(lowOffset, static_cast<T>(newCompensation));
                } else {
                    const float newValue = oldValue + lowValue;
                    xDiffLocal.SetValue(lowOffset, static_cast<T>(newValue));
                }
                minTouched = MinU64(minTouched, lowOffset);
                maxTouched = MaxU64(maxTouched, lowOffset);
            }

            if (xPoint.low != xPoint.high) {
                const float highWeight = xPoint.highWeight * yWeight;
                const float highValue = yDiffValue * highWeight;
                if (highValue == 0.0F) {
                    continue;
                }
                const uint64_t highOffset = static_cast<uint64_t>(xPoint.high);
                const float oldValue = static_cast<float>(xDiffLocal.GetValue(highOffset));
                if (useLocalCompensation) {
                    const float compensation = static_cast<float>(compLocal.GetValue(highOffset));
                    const float correctedValue = highValue - compensation;
                    const float newValue = oldValue + correctedValue;
                    const float newCompensation = (newValue - oldValue) - correctedValue;
                    xDiffLocal.SetValue(highOffset, static_cast<T>(newValue));
                    compLocal.SetValue(highOffset, static_cast<T>(newCompensation));
                } else {
                    const float newValue = oldValue + highValue;
                    xDiffLocal.SetValue(highOffset, static_cast<T>(newValue));
                }
                minTouched = MinU64(minTouched, highOffset);
                maxTouched = MaxU64(maxTouched, highOffset);
            }
        }

        if (minTouched < xDiffW_) {
            PipeBarrier<PIPE_ALL>();
            for (uint64_t xOffset = minTouched; xOffset <= maxTouched; ++xOffset) {
                const T value = xDiffLocal.GetValue(xOffset);
                if (static_cast<float>(value) != 0.0F) {
                    AtomicAddScalar(value, xDiffRowOffset + xOffset);
                    xDiffLocal.SetValue(xOffset, static_cast<T>(0));
                    if (useLocalCompensation) {
                        compLocal.SetValue(xOffset, static_cast<T>(0));
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ZeroNdOutputChannel(uint64_t xDiffBase)
{
    const uint64_t channelElemNum = xDiffH_ * xDiffW_;
    if (channelElemNum == 0U) {
        return;
    }

    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    const uint64_t zeroElemNum = MinU64(kNdZeroBufferElemNum, channelElemNum);
    Duplicate(xDiffLocal, static_cast<T>(0), static_cast<int32_t>(zeroElemNum));
    PipeBarrier<PIPE_ALL>();

    for (uint64_t offset = 0U; offset < channelElemNum; offset += kNdZeroBufferElemNum) {
        const uint64_t copyElemNum = MinU64(kNdZeroBufferElemNum, channelElemNum - offset);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyElemNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(xDiffGm_[xDiffBase + offset], xDiffLocal, copyParams);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessNdOutputChannel(uint64_t nIdx, uint64_t cIdx)
{
    if (nIdx >= xDiffN_ || cIdx >= xDiffC_) {
        return;
    }

    const uint64_t outputPlaneElemNum = xDiffH_ * xDiffW_;
    const uint64_t xDiffBase = (nIdx * c1_ + cIdx) * outputPlaneElemNum;
    ZeroNdOutputChannel(xDiffBase);

    const bool useForwardRoiOrder = roiEndMode_ == 0 && c1_ < 80U;
    for (uint64_t roiLoop = 0U; roiLoop < roiCount_; ++roiLoop) {
        const uint64_t roiIndex = useForwardRoiOrder ? roiLoop : (roiCount_ - 1U - roiLoop);
        RoiBox roi;
        LoadRoi(roiIndex, roi);

        const int32_t fmIdx = FloorToInt(roi.batchIdx);
        if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) != nIdx) {
            continue;
        }

        float roiStartX = roi.startX * spatialScale_;
        float roiStartY = roi.startY * spatialScale_;
        float roiEndX = roi.endX * spatialScale_;
        float roiEndY = roi.endY * spatialScale_;

        if (roiEndMode_ == 1 || roiEndMode_ == 3) {
            roiEndX += spatialScale_;
            roiEndY += spatialScale_;
        } else if (roiEndMode_ == 2) {
            roiStartX -= 0.5F;
            roiStartY -= 0.5F;
            roiEndX -= 0.5F;
            roiEndY -= 0.5F;
        }

        float roiWidth = roiEndX - roiStartX;
        float roiHeight = roiEndY - roiStartY;
        if (roiEndMode_ < 2) {
            roiWidth = MaxF32(roiWidth, 1.0F);
            roiHeight = MaxF32(roiHeight, 1.0F);
        } else if (roiEndMode_ == 3) {
            roiWidth = MaxF32(roiWidth, 0.0F);
            roiHeight = MaxF32(roiHeight, 0.0F);
        }

        const float binSizeW = roiWidth * pooledWidthReciprocal_;
        const float binSizeH = roiHeight * pooledHeightReciprocal_;
        const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
        const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
        if (sampleW <= 0 || sampleH <= 0) {
            continue;
        }

        const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) :
                                             (binSizeW / static_cast<float>(sampleW));
        const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) :
                                             (binSizeH / static_cast<float>(sampleH));

        for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
            for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                    continue;
                }

                float lowWeight = yPoint.lowWeight;
                if (yPoint.high == yPoint.low) {
                    lowWeight += yPoint.highWeight;
                }
                const uint64_t lowRowOffset = xDiffBase + static_cast<uint64_t>(yPoint.low) * xDiffW_;
                AccumulateNdOutputRow(roiIndex, cIdx, poolH, lowRowOffset, lowWeight, roiStartX, binSizeW, gridW,
                                      sampleW);

                if (yPoint.high != yPoint.low) {
                    const uint64_t highRowOffset = xDiffBase + static_cast<uint64_t>(yPoint.high) * xDiffW_;
                    AccumulateNdOutputRow(roiIndex, cIdx, poolH, highRowOffset, yPoint.highWeight, roiStartX, binSizeW,
                                          gridW, sampleW);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessOneRoiMoveOneRowSumInUb(uint64_t roiIndex, uint64_t c1Start,
                                                                       uint64_t c1Num)
{
    RoiBox roi;
    LoadRoi(roiIndex, roi);

    const int32_t fmIdx = FloorToInt(roi.batchIdx);
    if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) >= xDiffN_) {
        return;
    }

    float roiStartX = roi.startX * spatialScale_;
    float roiStartY = roi.startY * spatialScale_;
    float roiEndX = roi.endX * spatialScale_;
    float roiEndY = roi.endY * spatialScale_;

    if (roiEndMode_ == 1 || roiEndMode_ == 3) {
        roiEndX += spatialScale_;
        roiEndY += spatialScale_;
    } else if (roiEndMode_ == 2) {
        roiStartX -= 0.5F;
        roiStartY -= 0.5F;
        roiEndX -= 0.5F;
        roiEndY -= 0.5F;
    }

    float roiWidth = roiEndX - roiStartX;
    float roiHeight = roiEndY - roiStartY;
    if (roiEndMode_ < 2) {
        roiWidth = MaxF32(roiWidth, 1.0F);
        roiHeight = MaxF32(roiHeight, 1.0F);
    } else if (roiEndMode_ == 3) {
        roiWidth = MaxF32(roiWidth, 0.0F);
        roiHeight = MaxF32(roiHeight, 0.0F);
    }

    const float binSizeW = roiWidth * pooledWidthReciprocal_;
    const float binSizeH = roiHeight * pooledHeightReciprocal_;
    const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
    const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
    if (sampleW <= 0 || sampleH <= 0) {
        return;
    }

    const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) : (binSizeW / static_cast<float>(sampleW));
    const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) : (binSizeH / static_cast<float>(sampleH));
    if (NoSampleOverlap(roiStartX, roiStartY, gridW, gridH, sampleW, sampleH)) {
        return;
    }

    const uint32_t yDiffRowElemNum = static_cast<uint32_t>(pooledWidth_) * static_cast<uint32_t>(kC0Size);
    const uint32_t yDiffRowBytes = yDiffRowElemNum * static_cast<uint32_t>(sizeof(T));

    // 第3级 CSE：X 采样点只依赖 roi，与 yIndex 无关。这里预计算一次（存 low/high/lw/hw），
    // 供下面 poolH×sampleH×角 循环复用，避免每次重算 CalcGridCoordinate/CalcAxisPoint。
    // 本循环原本就要遍历所有 X 采样点算 activeXMin/Max，顺便存储几乎零额外开销。
    LocalTensor<int32_t> xIdxLocal = xIdxBuf_.template Get<int32_t>();
    LocalTensor<float> xWeightLocal = xWeightBuf_.template Get<float>();
    int32_t activeXMin = static_cast<int32_t>(xDiffW_);
    int32_t activeXMax = -1;
    // 第5级（批量散射）预合并：同一 (poolW,target) 的采样点 X 权重预累加。
    // X 权重和 与 yIndex 无关（Sigma(yWeight*w_s)=yWeight*Sigma w_s），故可在此一次性合并。
    // 合并只在同 poolW 组内（src=yDiffRow[poolW] 相同才能合并）。条目布局：
    // xIdxLocal 2 int/条 [poolW,target]；xWeightLocal 1 float/条 [mergedWeight]。
    int32_t xSampleCount = 0; // 合并后条目数
    for (int32_t poolW = 0; poolW < pooledWidth_; ++poolW) {
        const int32_t groupStart = xSampleCount;
        for (int32_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
            const bool lowActive = (xPoint.lowWeight != 0.0F ||
                                    (xPoint.low == xPoint.high && xPoint.highWeight != 0.0F));
            const bool highActive = (xPoint.low != xPoint.high && xPoint.highWeight != 0.0F);
            if (!lowActive && !highActive) {
                continue;
            }
            if (lowActive) {
                float lw = xPoint.lowWeight;
                if (xPoint.low == xPoint.high) {
                    lw += xPoint.highWeight;
                }
                int32_t hit = -1;
                for (int32_t k = groupStart; k < xSampleCount; ++k) {
                    if (xIdxLocal.GetValue(k * 2 + 1) == xPoint.low) {
                        hit = k;
                        break;
                    }
                }
                if (hit >= 0) {
                    xWeightLocal.SetValue(hit, xWeightLocal.GetValue(hit) + lw);
                } else {
                    xIdxLocal.SetValue(xSampleCount * 2, poolW);
                    xIdxLocal.SetValue(xSampleCount * 2 + 1, xPoint.low);
                    xWeightLocal.SetValue(xSampleCount, lw);
                    ++xSampleCount;
                    activeXMin = activeXMin < xPoint.low ? activeXMin : xPoint.low;
                    activeXMax = activeXMax > xPoint.low ? activeXMax : xPoint.low;
                }
            }
            if (highActive) {
                int32_t hit = -1;
                for (int32_t k = groupStart; k < xSampleCount; ++k) {
                    if (xIdxLocal.GetValue(k * 2 + 1) == xPoint.high) {
                        hit = k;
                        break;
                    }
                }
                if (hit >= 0) {
                    xWeightLocal.SetValue(hit, xWeightLocal.GetValue(hit) + xPoint.highWeight);
                } else {
                    xIdxLocal.SetValue(xSampleCount * 2, poolW);
                    xIdxLocal.SetValue(xSampleCount * 2 + 1, xPoint.high);
                    xWeightLocal.SetValue(xSampleCount, xPoint.highWeight);
                    ++xSampleCount;
                    activeXMin = activeXMin < xPoint.high ? activeXMin : xPoint.high;
                    activeXMax = activeXMax > xPoint.high ? activeXMax : xPoint.high;
                }
            }
        }
    }
    if (activeXMax < activeXMin) {
        return;
    }

    for (uint64_t c1BatchOffset = 0U; c1BatchOffset < c1Num; c1BatchOffset += c1BatchMax_) {
        const uint64_t innerC1Num = MinU64(c1BatchMax_, c1Num - c1BatchOffset);

        for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
            for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                const uint64_t yDiffOffset = ((roiIndex * c1_ + c1Idx) * static_cast<uint64_t>(pooledHeight_) +
                                              static_cast<uint64_t>(poolH)) *
                                             static_cast<uint64_t>(pooledWidth_) * kC0Size;
                LocalTensor<T> yDiffRowLocal = yDiffBuf_.template GetWithOffset<T>(
                    yDiffRowElemNum, static_cast<uint32_t>(innerOffset) * yDiffRowBytes);
                DataCopy(yDiffRowLocal, yDiffGm_[yDiffOffset], yDiffRowElemNum);
            }
            PipeBarrier<PIPE_ALL>();

            // ===== yIndex 归并优化 =====
            // 实测(Test_020)：一个 poolH 内 sampleH×2=16 次贡献只命中 ~3.6 个不同输出行 yIndex，
            // 77.5% 的调用在重复写同一行，各自重跑 Duplicate+散射+同步+atomic写回。
            // 同一 yIndex 的多次命中，X 采样结构(poolW,target,xWeight)完全相同，仅 yWeight 不同：
            //   Σ贡献 = Σ(yWeight) × (xWeight × yDiffRow[poolW])  —— 分配律，精确等价。
            // 归并表与 c1 无关(yPoint 只依赖 poolH/gridHIdx)，构建一次、跨 c1 复用。
            int32_t yHitIdx[kMaxYHits];
            float yHitWeight[kMaxYHits];
            int32_t yHitCount = 0;
            bool yMergeOverflow = false;
            for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                    continue;
                }
                float lowWeight = yPoint.lowWeight;
                if (yPoint.low == yPoint.high) {
                    lowWeight += yPoint.highWeight;
                }
                MergeYHit(yHitIdx, yHitWeight, yHitCount, yPoint.low, lowWeight, yMergeOverflow);
                if (yPoint.low != yPoint.high) {
                    MergeYHit(yHitIdx, yHitWeight, yHitCount, yPoint.high, yPoint.highWeight, yMergeOverflow);
                }
            }
            // 防御：万一溢出(现实不触及)，回退原逐点全量路径，保证正确。
            if (yMergeOverflow) {
                yHitCount = 0;
                for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                    const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                    const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                    if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                        continue;
                    }
                    float lowWeight = yPoint.lowWeight;
                    if (yPoint.low == yPoint.high) {
                        lowWeight += yPoint.highWeight;
                    }
                    for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                        const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                        const uint64_t xDiffBase = ((static_cast<uint64_t>(fmIdx) * c1_ + c1Idx) * xDiffH_ * xDiffW_) *
                                                   kC0Size;
                        LocalTensor<T> yDiffRowLocal = yDiffBuf_.template GetWithOffset<T>(
                            yDiffRowElemNum, static_cast<uint32_t>(innerOffset) * yDiffRowBytes);
                        AccumulateMoveOneRowOutputRow(yDiffRowLocal, xDiffBase, yPoint.low, lowWeight, xSampleCount,
                                                      activeXMin, activeXMax);
                        if (yPoint.low != yPoint.high) {
                            AccumulateMoveOneRowOutputRow(yDiffRowLocal, xDiffBase, yPoint.high, yPoint.highWeight,
                                                          xSampleCount, activeXMin, activeXMax);
                        }
                    }
                }
            }

            // 对每个唯一 yIndex（跨 c1）只散射+写回一次
            for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                const uint64_t xDiffBase = ((static_cast<uint64_t>(fmIdx) * c1_ + c1Idx) * xDiffH_ * xDiffW_) * kC0Size;
                LocalTensor<T> yDiffRowLocal = yDiffBuf_.template GetWithOffset<T>(
                    yDiffRowElemNum, static_cast<uint32_t>(innerOffset) * yDiffRowBytes);
                for (int32_t k = 0; k < yHitCount; ++k) {
                    AccumulateMoveOneRowOutputRow(yDiffRowLocal, xDiffBase, yHitIdx[k], yHitWeight[k], xSampleCount,
                                                  activeXMin, activeXMax);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessOneRoi(uint64_t roiIndex, uint64_t c1Start, uint64_t c1Num)
{
    RoiBox roi;
    LoadRoi(roiIndex, roi);

    const int32_t fmIdx = FloorToInt(roi.batchIdx);
    if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) >= xDiffN_) {
        return;
    }

    float roiStartX = roi.startX * spatialScale_;
    float roiStartY = roi.startY * spatialScale_;
    float roiEndX = roi.endX * spatialScale_;
    float roiEndY = roi.endY * spatialScale_;

    if (roiEndMode_ == 1 || roiEndMode_ == 3) {
        roiEndX += spatialScale_;
        roiEndY += spatialScale_;
    } else if (roiEndMode_ == 2) {
        roiStartX -= 0.5F;
        roiStartY -= 0.5F;
        roiEndX -= 0.5F;
        roiEndY -= 0.5F;
    }

    float roiWidth = roiEndX - roiStartX;
    float roiHeight = roiEndY - roiStartY;
    if (roiEndMode_ < 2) {
        roiWidth = MaxF32(roiWidth, 1.0F);
        roiHeight = MaxF32(roiHeight, 1.0F);
    } else if (roiEndMode_ == 3) {
        roiWidth = MaxF32(roiWidth, 0.0F);
        roiHeight = MaxF32(roiHeight, 0.0F);
    }

    const float binSizeW = roiWidth * pooledWidthReciprocal_;
    const float binSizeH = roiHeight * pooledHeightReciprocal_;
    const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
    const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
    if (sampleW <= 0 || sampleH <= 0) {
        return;
    }

    const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) : (binSizeW / static_cast<float>(sampleW));
    const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) : (binSizeH / static_cast<float>(sampleH));
    if (NoSampleOverlap(roiStartX, roiStartY, gridW, gridH, sampleW, sampleH)) {
        return;
    }

    const uint32_t yDiffSliceElemNum = static_cast<uint32_t>(kC0Size);
    const uint32_t yDiffSliceBytes = static_cast<uint32_t>(kC0Size * sizeof(T));
    const uint32_t yDiffRowElemNum = static_cast<uint32_t>(pooledWidth_) * yDiffSliceElemNum;
    const uint32_t yDiffRowBytes = yDiffRowElemNum * static_cast<uint32_t>(sizeof(T));
    const bool useMoveOneRow = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW;
    const bool useNc1hwc0RoiSplit = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;

    for (uint64_t c1BatchOffset = 0U; c1BatchOffset < c1Num; c1BatchOffset += c1BatchMax_) {
        const uint64_t innerC1Num = MinU64(c1BatchMax_, c1Num - c1BatchOffset);

        if (useMoveOneRow || useNc1hwc0RoiSplit) {
            for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
                for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                    const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                    const uint64_t yDiffOffset = ((roiIndex * c1_ + c1Idx) * static_cast<uint64_t>(pooledHeight_) +
                                                  static_cast<uint64_t>(poolH)) *
                                                 static_cast<uint64_t>(pooledWidth_) * kC0Size;
                    LocalTensor<T> yDiffRowLocal = yDiffBuf_.template GetWithOffset<T>(
                        yDiffRowElemNum, static_cast<uint32_t>(innerOffset) * yDiffRowBytes);
                    DataCopy(yDiffRowLocal, yDiffGm_[yDiffOffset], yDiffRowElemNum);
                }
                PipeBarrier<PIPE_ALL>();

                for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                    const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                    const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                    if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                        continue;
                    }

                    for (int32_t poolW = 0; poolW < pooledWidth_; ++poolW) {
                        for (int32_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
                            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
                            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
                            if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                                continue;
                            }

                            const float w1 = yPoint.lowWeight * xPoint.lowWeight;
                            const float w2 = yPoint.lowWeight * xPoint.highWeight;
                            const float w3 = yPoint.highWeight * xPoint.lowWeight;
                            const float w4 = yPoint.highWeight * xPoint.highWeight;

                            for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                                const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                                const uint64_t xDiffBase = ((static_cast<uint64_t>(fmIdx) * c1_ + c1Idx) * xDiffH_ *
                                                            xDiffW_) *
                                                           kC0Size;
                                const uint64_t xDiffOffset1 = xDiffBase + (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                                           static_cast<uint64_t>(xPoint.low)) *
                                                                              kC0Size;
                                const uint64_t xDiffOffset2 = xDiffBase + (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                                           static_cast<uint64_t>(xPoint.high)) *
                                                                              kC0Size;
                                const uint64_t xDiffOffset3 = xDiffBase +
                                                              (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                               static_cast<uint64_t>(xPoint.low)) *
                                                                  kC0Size;
                                const uint64_t xDiffOffset4 = xDiffBase +
                                                              (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                               static_cast<uint64_t>(xPoint.high)) *
                                                                  kC0Size;

                                LocalTensor<T> yDiffLocal = yDiffBuf_.template GetWithOffset<T>(
                                    yDiffSliceElemNum, static_cast<uint32_t>(innerOffset) * yDiffRowBytes +
                                                           static_cast<uint32_t>(poolW) * yDiffSliceBytes);
                                AtomicAddVectorActive(yDiffLocal, xDiffOffset1, w1);
                                AtomicAddVectorActive(yDiffLocal, xDiffOffset2, w2);
                                AtomicAddVectorActive(yDiffLocal, xDiffOffset3, w3);
                                AtomicAddVectorActive(yDiffLocal, xDiffOffset4, w4);
                            }
                        }
                    }
                }
            }
        } else {
            for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
                for (int32_t poolW = 0; poolW < pooledWidth_; ++poolW) {
                    for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                        const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                        const uint64_t yDiffOffset = ((roiIndex * c1_ + c1Idx) * static_cast<uint64_t>(pooledHeight_) *
                                                          static_cast<uint64_t>(pooledWidth_) +
                                                      static_cast<uint64_t>(poolH) *
                                                          static_cast<uint64_t>(pooledWidth_) +
                                                      static_cast<uint64_t>(poolW)) *
                                                     kC0Size;
                        LocalTensor<T> yDiffLocal = yDiffBuf_.template GetWithOffset<T>(
                            yDiffSliceElemNum, static_cast<uint32_t>(innerOffset) * yDiffSliceBytes);
                        DataCopy(yDiffLocal, yDiffGm_[yDiffOffset], yDiffSliceElemNum);
                    }
                    PipeBarrier<PIPE_ALL>();

                    for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                        const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                        const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                        if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                            continue;
                        }

                        for (int32_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
                            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
                            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
                            if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                                continue;
                            }

                            const float w1 = yPoint.lowWeight * xPoint.lowWeight;
                            const float w2 = yPoint.lowWeight * xPoint.highWeight;
                            const float w3 = yPoint.highWeight * xPoint.lowWeight;
                            const float w4 = yPoint.highWeight * xPoint.highWeight;

                            for (uint64_t innerOffset = 0U; innerOffset < innerC1Num; ++innerOffset) {
                                const uint64_t c1Idx = c1Start + c1BatchOffset + innerOffset;
                                const uint64_t xDiffBase = ((static_cast<uint64_t>(fmIdx) * c1_ + c1Idx) * xDiffH_ *
                                                            xDiffW_) *
                                                           kC0Size;
                                const uint64_t xDiffOffset1 = xDiffBase + (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                                           static_cast<uint64_t>(xPoint.low)) *
                                                                              kC0Size;
                                const uint64_t xDiffOffset2 = xDiffBase + (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                                           static_cast<uint64_t>(xPoint.high)) *
                                                                              kC0Size;
                                const uint64_t xDiffOffset3 = xDiffBase +
                                                              (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                               static_cast<uint64_t>(xPoint.low)) *
                                                                  kC0Size;
                                const uint64_t xDiffOffset4 = xDiffBase +
                                                              (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                               static_cast<uint64_t>(xPoint.high)) *
                                                                  kC0Size;

                                LocalTensor<T> yDiffLocal = yDiffBuf_.template GetWithOffset<T>(
                                    yDiffSliceElemNum, static_cast<uint32_t>(innerOffset) * yDiffSliceBytes);
                                AtomicAddVector(yDiffLocal, xDiffOffset1, w1);
                                AtomicAddVector(yDiffLocal, xDiffOffset2, w2);
                                AtomicAddVector(yDiffLocal, xDiffOffset3, w3);
                                AtomicAddVector(yDiffLocal, xDiffOffset4, w4);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessOneRoiNc1hwc0Scalar(uint64_t roiIndex, uint64_t cStart, uint64_t cNum)
{
    RoiBox roi;
    LoadRoi(roiIndex, roi);

    const int32_t fmIdx = FloorToInt(roi.batchIdx);
    if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) >= xDiffN_) {
        return;
    }

    float roiStartX = roi.startX * spatialScale_;
    float roiStartY = roi.startY * spatialScale_;
    float roiEndX = roi.endX * spatialScale_;
    float roiEndY = roi.endY * spatialScale_;

    if (roiEndMode_ == 1 || roiEndMode_ == 3) {
        roiEndX += spatialScale_;
        roiEndY += spatialScale_;
    } else if (roiEndMode_ == 2) {
        roiStartX -= 0.5F;
        roiStartY -= 0.5F;
        roiEndX -= 0.5F;
        roiEndY -= 0.5F;
    }

    float roiWidth = roiEndX - roiStartX;
    float roiHeight = roiEndY - roiStartY;
    if (roiEndMode_ < 2) {
        roiWidth = MaxF32(roiWidth, 1.0F);
        roiHeight = MaxF32(roiHeight, 1.0F);
    } else if (roiEndMode_ == 3) {
        roiWidth = MaxF32(roiWidth, 0.0F);
        roiHeight = MaxF32(roiHeight, 0.0F);
    }

    const float binSizeW = roiWidth * pooledWidthReciprocal_;
    const float binSizeH = roiHeight * pooledHeightReciprocal_;
    const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
    const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
    if (sampleW <= 0 || sampleH <= 0) {
        return;
    }

    const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) : (binSizeW / static_cast<float>(sampleW));
    const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) : (binSizeH / static_cast<float>(sampleH));

    const uint64_t cEnd = MinU64(cStart + cNum, xDiffC_);
    for (uint64_t channelIdx = cStart; channelIdx < cEnd;) {
        const uint64_t c1Idx = channelIdx / kC0Size;
        const uint64_t c0Start = channelIdx % kC0Size;
        const uint64_t validC0 = MinU64(kC0Size - c0Start, cEnd - channelIdx);

        const uint64_t xDiffBase = ((static_cast<uint64_t>(fmIdx) * c1_ + c1Idx) * xDiffH_ * xDiffW_) * kC0Size;
        const uint64_t xDiffPlaneElemNum = xDiffH_ * xDiffW_ * kC0Size;
        LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
        LocalTensor<T> yDiffLocal = yDiffBuf_.template Get<T>();
        DataCopy(xDiffLocal, xDiffGm_[xDiffBase], static_cast<uint32_t>(xDiffPlaneElemNum));
        PipeBarrier<PIPE_ALL>();

        for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
            for (int32_t poolW = 0; poolW < pooledWidth_; ++poolW) {
                const uint64_t yDiffOffset = (((roiIndex * c1_ + c1Idx) * static_cast<uint64_t>(pooledHeight_) +
                                               static_cast<uint64_t>(poolH)) *
                                                  static_cast<uint64_t>(pooledWidth_) +
                                              static_cast<uint64_t>(poolW)) *
                                             kC0Size;
                DataCopy(yDiffLocal, yDiffGm_[yDiffOffset], static_cast<uint32_t>(kC0Size));
                PipeBarrier<PIPE_ALL>();

                for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                    const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                    const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                    if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                        continue;
                    }

                    for (int32_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
                        const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
                        const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
                        if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                            continue;
                        }

                        const float w1 = yPoint.lowWeight * xPoint.lowWeight;
                        const float w2 = yPoint.lowWeight * xPoint.highWeight;
                        const float w3 = yPoint.highWeight * xPoint.lowWeight;
                        const float w4 = yPoint.highWeight * xPoint.highWeight;

                        const uint64_t xDiffOffset1 = (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                       static_cast<uint64_t>(xPoint.low)) *
                                                      kC0Size;
                        const uint64_t xDiffOffset2 = (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                       static_cast<uint64_t>(xPoint.high)) *
                                                      kC0Size;
                        const uint64_t xDiffOffset3 = (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                       static_cast<uint64_t>(xPoint.low)) *
                                                      kC0Size;
                        const uint64_t xDiffOffset4 = (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                       static_cast<uint64_t>(xPoint.high)) *
                                                      kC0Size;

                        (void)c0Start;
                        (void)validC0;
                        AccumulateLocalVector(yDiffLocal, xDiffOffset1, w1);
                        AccumulateLocalVector(yDiffLocal, xDiffOffset2, w2);
                        AccumulateLocalVector(yDiffLocal, xDiffOffset3, w3);
                        AccumulateLocalVector(yDiffLocal, xDiffOffset4, w4);
                    }
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(xDiffGm_[xDiffBase], xDiffLocal, static_cast<uint32_t>(xDiffPlaneElemNum));
        PipeBarrier<PIPE_ALL>();
        channelIdx += validC0;
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessC1PlaneNc1hwc0(uint64_t c1Idx)
{
    const uint64_t xDiffPlaneElemNum = xDiffH_ * xDiffW_ * kC0Size;
    LocalTensor<T> xDiffLocal = xDiffBuf_.template Get<T>();
    LocalTensor<T> yDiffLocal = yDiffBuf_.template Get<T>();

    for (uint64_t nIdx = 0U; nIdx < xDiffN_; ++nIdx) {
        Duplicate(xDiffLocal, static_cast<T>(0), static_cast<int32_t>(xDiffPlaneElemNum));
        PipeBarrier<PIPE_ALL>();

        for (uint64_t roiIndex = 0U; roiIndex < roiCount_; ++roiIndex) {
            RoiBox roi;
            LoadRoi(roiIndex, roi);

            const int32_t fmIdx = FloorToInt(roi.batchIdx);
            if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) != nIdx) {
                continue;
            }

            float roiStartX = roi.startX * spatialScale_;
            float roiStartY = roi.startY * spatialScale_;
            float roiEndX = roi.endX * spatialScale_;
            float roiEndY = roi.endY * spatialScale_;

            if (roiEndMode_ == 1 || roiEndMode_ == 3) {
                roiEndX += spatialScale_;
                roiEndY += spatialScale_;
            } else if (roiEndMode_ == 2) {
                roiStartX -= 0.5F;
                roiStartY -= 0.5F;
                roiEndX -= 0.5F;
                roiEndY -= 0.5F;
            }

            float roiWidth = roiEndX - roiStartX;
            float roiHeight = roiEndY - roiStartY;
            if (roiEndMode_ < 2) {
                roiWidth = MaxF32(roiWidth, 1.0F);
                roiHeight = MaxF32(roiHeight, 1.0F);
            } else if (roiEndMode_ == 3) {
                roiWidth = MaxF32(roiWidth, 0.0F);
                roiHeight = MaxF32(roiHeight, 0.0F);
            }

            const float binSizeW = roiWidth * pooledWidthReciprocal_;
            const float binSizeH = roiHeight * pooledHeightReciprocal_;
            const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
            const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
            if (sampleW <= 0 || sampleH <= 0) {
                continue;
            }

            const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) :
                                                 (binSizeW / static_cast<float>(sampleW));
            const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) :
                                                 (binSizeH / static_cast<float>(sampleH));

            for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
                for (int32_t poolW = 0; poolW < pooledWidth_; ++poolW) {
                    const uint64_t yDiffOffset = (((roiIndex * c1_ + c1Idx) * static_cast<uint64_t>(pooledHeight_) +
                                                   static_cast<uint64_t>(poolH)) *
                                                      static_cast<uint64_t>(pooledWidth_) +
                                                  static_cast<uint64_t>(poolW)) *
                                                 kC0Size;
                    DataCopy(yDiffLocal, yDiffGm_[yDiffOffset], static_cast<uint32_t>(kC0Size));
                    PipeBarrier<PIPE_ALL>();

                    for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                        const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                        const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                        if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                            continue;
                        }

                        for (int32_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
                            const float x = CalcGridCoordinate(roiStartX, gridW, sampleW, poolW, gridWIdx);
                            const AxisPoint xPoint = CalcAxisPoint(x, static_cast<int32_t>(xDiffW_), sampleW);
                            if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                                continue;
                            }

                            const float w1 = yPoint.lowWeight * xPoint.lowWeight;
                            const float w2 = yPoint.lowWeight * xPoint.highWeight;
                            const float w3 = yPoint.highWeight * xPoint.lowWeight;
                            const float w4 = yPoint.highWeight * xPoint.highWeight;

                            const uint64_t xDiffOffset1 = (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                           static_cast<uint64_t>(xPoint.low)) *
                                                          kC0Size;
                            const uint64_t xDiffOffset2 = (static_cast<uint64_t>(yPoint.low) * xDiffW_ +
                                                           static_cast<uint64_t>(xPoint.high)) *
                                                          kC0Size;
                            const uint64_t xDiffOffset3 = (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                           static_cast<uint64_t>(xPoint.low)) *
                                                          kC0Size;
                            const uint64_t xDiffOffset4 = (static_cast<uint64_t>(yPoint.high) * xDiffW_ +
                                                           static_cast<uint64_t>(xPoint.high)) *
                                                          kC0Size;

                            AccumulateLocalVector(yDiffLocal, xDiffOffset1, w1);
                            AccumulateLocalVector(yDiffLocal, xDiffOffset2, w2);
                            AccumulateLocalVector(yDiffLocal, xDiffOffset3, w3);
                            AccumulateLocalVector(yDiffLocal, xDiffOffset4, w4);
                        }
                    }
                }
            }
        }

        const uint64_t xDiffBase = ((nIdx * c1_ + c1Idx) * xDiffH_ * xDiffW_) * kC0Size;
        PipeBarrier<PIPE_ALL>();
        DataCopy(xDiffGm_[xDiffBase], xDiffLocal, static_cast<uint32_t>(xDiffPlaneElemNum));
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::ProcessOneRoiNd(uint64_t roiIndex, uint64_t cStart, uint64_t cNum)
{
    RoiBox roi;
    LoadRoi(roiIndex, roi);

    const int32_t fmIdx = FloorToInt(roi.batchIdx);
    if (fmIdx < 0 || static_cast<uint64_t>(fmIdx) >= xDiffN_) {
        return;
    }

    float roiStartX = roi.startX * spatialScale_;
    float roiStartY = roi.startY * spatialScale_;
    float roiEndX = roi.endX * spatialScale_;
    float roiEndY = roi.endY * spatialScale_;

    if (roiEndMode_ == 1 || roiEndMode_ == 3) {
        roiEndX += spatialScale_;
        roiEndY += spatialScale_;
    } else if (roiEndMode_ == 2) {
        roiStartX -= 0.5F;
        roiStartY -= 0.5F;
        roiEndX -= 0.5F;
        roiEndY -= 0.5F;
    }

    float roiWidth = roiEndX - roiStartX;
    float roiHeight = roiEndY - roiStartY;
    if (roiEndMode_ < 2) {
        roiWidth = MaxF32(roiWidth, 1.0F);
        roiHeight = MaxF32(roiHeight, 1.0F);
    } else if (roiEndMode_ == 3) {
        roiWidth = MaxF32(roiWidth, 0.0F);
        roiHeight = MaxF32(roiHeight, 0.0F);
    }

    const float binSizeW = roiWidth * pooledWidthReciprocal_;
    const float binSizeH = roiHeight * pooledHeightReciprocal_;
    const int32_t sampleW = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeW);
    const int32_t sampleH = sampleNum_ > 0 ? sampleNum_ : CeilToInt(binSizeH);
    if (sampleW <= 0 || sampleH <= 0) {
        return;
    }

    const float gridW = sampleNum_ > 0 ? (binSizeW * sampleNumReciprocal_) : (binSizeW / static_cast<float>(sampleW));
    const float gridH = sampleNum_ > 0 ? (binSizeH * sampleNumReciprocal_) : (binSizeH / static_cast<float>(sampleH));

    for (uint64_t cOffset = 0U; cOffset < cNum; ++cOffset) {
        const uint64_t cIdx = cStart + cOffset;
        const uint64_t xDiffBase = (static_cast<uint64_t>(fmIdx) * c1_ + cIdx) * xDiffH_ * xDiffW_;

        for (int32_t poolH = 0; poolH < pooledHeight_; ++poolH) {
            for (int32_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                const float y = CalcGridCoordinate(roiStartY, gridH, sampleH, poolH, gridHIdx);
                const AxisPoint yPoint = CalcAxisPoint(y, static_cast<int32_t>(xDiffH_), sampleH);
                if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                    continue;
                }

                float lowWeight = yPoint.lowWeight;
                if (yPoint.high == yPoint.low) {
                    lowWeight += yPoint.highWeight;
                }
                const uint64_t lowRowOffset = xDiffBase + static_cast<uint64_t>(yPoint.low) * xDiffW_;
                AccumulateNdOutputRow(roiIndex, cIdx, poolH, lowRowOffset, lowWeight, roiStartX, binSizeW, gridW,
                                      sampleW);

                if (yPoint.high != yPoint.low) {
                    const uint64_t highRowOffset = xDiffBase + static_cast<uint64_t>(yPoint.high) * xDiffW_;
                    AccumulateNdOutputRow(roiIndex, cIdx, poolH, highRowOffset, yPoint.highWeight, roiStartX, binSizeW,
                                          gridW, sampleW);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void RoiAlignGrad<T>::Process()
{
    InitLocalBuffer();

    const bool useMoveOneRow = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW;
    const bool useNc1hwc0RoiSplit = tilingKey_ == ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;
    if (isNd_ == 0 && !useMoveOneRow && !useNc1hwc0RoiSplit) {
        if (coreNc_ == 0U) {
            return;
        }
        for (uint64_t c1Offset = 0U; c1Offset < coreNc_; ++c1Offset) {
            ProcessC1PlaneNc1hwc0(coreNcOffset_ + c1Offset);
        }
        return;
    }

    if (isNd_ != 0) {
        if (coreNc_ == 0U) {
            return;
        }
        for (uint64_t coreNIndex = 0U; coreNIndex < coreRoiCount_; ++coreNIndex) {
            uint64_t cStart = 0U;
            uint64_t cNum = 0U;
            CalcC1Range(coreNIndex, cStart, cNum);
            for (uint64_t cOffset = 0U; cOffset < cNum; ++cOffset) {
                ProcessNdOutputChannel(coreRoiOffset_ + coreNIndex, cStart + cOffset);
            }
        }
        return;
    }

    ZeroOutput();
    if (useNc1hwc0RoiSplit) {
        // 无参硬件 SyncAll 不需要 GM workspace，不再调用 ZeroSyncWorkspace
        SyncAfterZero();
    }

    if (coreNc_ == 0U && !useNc1hwc0RoiSplit) {
        return;
    }

    (void)tilingKey_;

    if (isNd_ == 0 && useNc1hwc0RoiSplit) {
        // 负载均衡：按全局 nc(=roi*c1) 索引 round-robin 分配工作项，核 b 处理
        // nc = b, b+coreNum, b+2*coreNum ...。相比连续分块，重 roi(采样点多、计算量
        // 大)的多个 c1 平面被拆散到不同核，避免单个大 roi 独占一核导致的负载不均。
        // 每次只处理 1 个 c1（c1Num=1），X 采样点预计算 per-(roi,c1) 重算，开销小。
        // atomic 写回已就位，拆分无写冲突，累加结果不变，精度零风险。
        SetAtomicAdd<T>();
        const uint64_t totalNc = roiCount_ * c1_;
        for (uint64_t nc = static_cast<uint64_t>(blockIdx_); nc < totalNc; nc += runningCoreNum_) {
            const uint64_t roiIndex = nc / c1_;
            const uint64_t c1Idx = nc % c1_;
            ProcessOneRoiMoveOneRowSumInUb(roiIndex, c1Idx, 1U);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        return;
    }

    if (isNd_ == 0 && useMoveOneRow) {
        SetAtomicAdd<T>();
        for (uint64_t roiIndex = 0U; roiIndex < roiCount_; ++roiIndex) {
            ProcessOneRoi(roiIndex, coreNcOffset_, coreNc_);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        return;
    }

    if (isNd_ != 0) {
        for (uint64_t coreRoiIndex = 0U; coreRoiIndex < coreRoiCount_; ++coreRoiIndex) {
            uint64_t c1Start = 0U;
            uint64_t c1Num = 0U;
            CalcC1Range(coreRoiIndex, c1Start, c1Num);
            if (c1Num == 0U) {
                continue;
            }
            ProcessOneRoiNd(coreRoiOffset_ + coreRoiIndex, c1Start, c1Num);
        }
        return;
    }

    for (uint64_t coreRoiIndex = 0U; coreRoiIndex < coreRoiCount_; ++coreRoiIndex) {
        uint64_t c1Start = 0U;
        uint64_t c1Num = 0U;
        CalcC1Range(coreRoiIndex, c1Start, c1Num);
        if (c1Num == 0U) {
            continue;
        }
        if (useNc1hwc0RoiSplit) {
            ProcessOneRoi(coreRoiOffset_ + coreRoiIndex, c1Start, c1Num);
        } else {
            ProcessOneRoiNc1hwc0Scalar(coreRoiOffset_ + coreRoiIndex, c1Start * kC0Size, c1Num * kC0Size);
        }
    }
}

} // namespace NsRoiAlignGrad

#endif // __ROI_ALIGN_GRAD_H__
