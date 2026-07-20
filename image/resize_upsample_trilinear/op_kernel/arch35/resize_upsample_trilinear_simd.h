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
 * \file resize_upsample_trilinear_simd.h
 * \brief D-only vector path for ResizeUpsampleTrilinear on arch35.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMD_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMD_H_

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

template <typename T>
struct DOnlyLocalTensors {
    LocalTensor<T> input0;
    LocalTensor<T> input1;
    LocalTensor<float> cachedFp0;
    LocalTensor<float> cachedFp1;
    LocalTensor<T> outputRaw;
    LocalTensor<float> outputFp;
};

struct DOnlyEvents {
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToV;
};

struct DOnlyCacheState {
    int64_t cached0 = -1;
    int64_t cached1 = -1;
};

template <typename T>
class ResizeUpsampleTrilinearSimd {
public:
    __aicore__ inline ResizeUpsampleTrilinearSimd() = default;
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    // Keep one output tile and explicitly synchronize its reuse. The two input
    // tiles cache adjacent D planes across output depths.
    static constexpr int64_t INPUT_TILE_ELEMENTS = std::is_same_v<T, float> ? 16384 : 8192;
    static constexpr int64_t OUTPUT_TILE_ELEMENTS = INPUT_TILE_ELEMENTS;
    static constexpr int64_t UB_BYTES = 2 * INPUT_TILE_ELEMENTS * sizeof(T) + OUTPUT_TILE_ELEMENTS * sizeof(T) +
                                        (std::is_same_v<T, float> ? 0 :
                                                                    (2 * INPUT_TILE_ELEMENTS + OUTPUT_TILE_ELEMENTS) *
                                                                        static_cast<int64_t>(sizeof(float)));
    static_assert(UB_BYTES <= 248 * 1024, "D-only buffers exceed A950 UB capacity");

    __aicore__ inline int32_t FindCachedSlot(int64_t index, int64_t cached0, int64_t cached1) const;
    __aicore__ inline void CopyIn(const LocalTensor<T>& dst, int64_t gmOffset, uint32_t count);
    __aicore__ inline void Compute(const LocalTensor<float>& src0, const LocalTensor<float>& src1,
                                   const LocalTensor<float>& outputFp, uint32_t localOffset, uint32_t count,
                                   float weight0, float weight1);
    __aicore__ inline void CastOutput(const LocalTensor<T>& outputRaw, const LocalTensor<float>& outputFp,
                                      uint32_t localOffset, uint32_t count);
    __aicore__ inline void CopyOut(const LocalTensor<T>& outputRaw, int64_t gmOffset, uint32_t count, event_t vToMte3,
                                   event_t mte3ToV);
    __aicore__ inline void InitLocalTensors(DOnlyLocalTensors<T>& tensors);
    __aicore__ inline void InitEvents(DOnlyEvents& events);
    __aicore__ inline bool GetNcRange(int64_t totalNc, int64_t& ncStart, int64_t& ncCount) const;
    __aicore__ inline void LoadDepthPair(DOnlyLocalTensors<T>& tensors, DOnlyCacheState& cache,
                                         const DOnlyEvents& events, int64_t inputNcOffset, int64_t planeElements,
                                         int64_t tileOffset, uint32_t count, int64_t in0, int64_t in1, int32_t& slot0,
                                         int32_t& slot1);
    __aicore__ inline uint32_t GetOutputsPerBatch(int64_t planeElements, int64_t tileOffset, uint32_t count) const;
    __aicore__ inline void ProcessTile(DOnlyLocalTensors<T>& tensors, const DOnlyEvents& events, int64_t inputNcOffset,
                                       int64_t outputNcOffset, int64_t planeElements, int64_t tileOffset,
                                       uint32_t count);
    __aicore__ inline void ProcessNc(DOnlyLocalTensors<T>& tensors, const DOnlyEvents& events, int64_t nc,
                                     int64_t planeElements, int64_t inputNcElements, int64_t outputNcElements);

private:
    const ResizeUpsampleTrilinearRegBaseTilingData* tilingData_ = nullptr;
    int32_t blockIdx_ = 0;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf0_;
    TBuf<TPosition::VECCALC> inputBuf1_;
    TBuf<TPosition::VECCALC> calcBuf0_;
    TBuf<TPosition::VECCALC> calcBuf1_;
    TBuf<TPosition::VECCALC> outputFpBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
};

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::Init(
    GM_ADDR input, GM_ADDR output, const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData)
{
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    pipe_.InitBuffer(inputBuf0_, INPUT_TILE_ELEMENTS * sizeof(T));
    pipe_.InitBuffer(inputBuf1_, INPUT_TILE_ELEMENTS * sizeof(T));
    if constexpr (!std::is_same_v<T, float>) {
        pipe_.InitBuffer(calcBuf0_, INPUT_TILE_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(calcBuf1_, INPUT_TILE_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(outputFpBuf_, OUTPUT_TILE_ELEMENTS * sizeof(float));
    }
    pipe_.InitBuffer(outputBuf_, OUTPUT_TILE_ELEMENTS * sizeof(T));
}

template <typename T>
__aicore__ inline int32_t ResizeUpsampleTrilinearSimd<T>::FindCachedSlot(int64_t index, int64_t cached0,
                                                                         int64_t cached1) const
{
    if (index == cached0) {
        return 0;
    }
    if (index == cached1) {
        return 1;
    }
    return -1;
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::CopyIn(const LocalTensor<T>& dst, int64_t gmOffset,
                                                              uint32_t count)
{
    DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(dst, inputGm_[gmOffset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::Compute(const LocalTensor<float>& src0,
                                                               const LocalTensor<float>& src1,
                                                               const LocalTensor<float>& outputFp, uint32_t localOffset,
                                                               uint32_t count, float weight0, float weight1)
{
    Muls(outputFp[localOffset], src1, weight1, count);
    PipeBarrier<PIPE_V>();
    Axpy(outputFp[localOffset], src0, weight0, count);
    PipeBarrier<PIPE_V>();

    // A zero self-Axpy preserves finite values and maps any Inf/NaN result to NaN, matching skipped H/W weights.
    Axpy(outputFp[localOffset], outputFp[localOffset], 0.0f, count);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::CastOutput(const LocalTensor<T>& outputRaw,
                                                                  const LocalTensor<float>& outputFp,
                                                                  uint32_t localOffset, uint32_t count)
{
    if constexpr (!std::is_same_v<T, float>) {
        if constexpr (std::is_same_v<T, half>) {
            Cast(outputRaw[localOffset], outputFp[localOffset], RoundMode::CAST_NONE, count);
        } else {
            Cast(outputRaw[localOffset], outputFp[localOffset], RoundMode::CAST_RINT, count);
        }
    }
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::CopyOut(const LocalTensor<T>& outputRaw, int64_t gmOffset,
                                                               uint32_t count, event_t vToMte3, event_t mte3ToV)
{
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm_[gmOffset], outputRaw, copyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::InitLocalTensors(DOnlyLocalTensors<T>& tensors)
{
    tensors.input0 = inputBuf0_.Get<T>();
    tensors.input1 = inputBuf1_.Get<T>();
    tensors.outputRaw = outputBuf_.Get<T>();
    if constexpr (std::is_same_v<T, float>) {
        tensors.cachedFp0 = tensors.input0;
        tensors.cachedFp1 = tensors.input1;
        tensors.outputFp = tensors.outputRaw;
    } else {
        tensors.cachedFp0 = calcBuf0_.Get<float>();
        tensors.cachedFp1 = calcBuf1_.Get<float>();
        tensors.outputFp = outputFpBuf_.Get<float>();
    }
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::InitEvents(DOnlyEvents& events)
{
    events.vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    events.mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    events.vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    events.mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

template <typename T>
__aicore__ inline bool ResizeUpsampleTrilinearSimd<T>::GetNcRange(int64_t totalNc, int64_t& ncStart,
                                                                  int64_t& ncCount) const
{
    int64_t coreNum = static_cast<int64_t>(GetBlockNum());
    if (blockIdx_ >= coreNum || coreNum <= 0) {
        return false;
    }
    int64_t ncBase = totalNc / coreNum;
    int64_t tailBlockNum = totalNc % coreNum;
    ncCount = ncBase + (blockIdx_ < tailBlockNum ? 1 : 0);
    ncStart = static_cast<int64_t>(blockIdx_) * ncBase + min(static_cast<int64_t>(blockIdx_), tailBlockNum);
    return ncCount > 0;
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::LoadDepthPair(DOnlyLocalTensors<T>& tensors,
                                                                     DOnlyCacheState& cache, const DOnlyEvents& events,
                                                                     int64_t inputNcOffset, int64_t planeElements,
                                                                     int64_t tileOffset, uint32_t count, int64_t in0,
                                                                     int64_t in1, int32_t& slot0, int32_t& slot1)
{
    slot0 = FindCachedSlot(in0, cache.cached0, cache.cached1);
    slot1 = FindCachedSlot(in1, cache.cached0, cache.cached1);
    uint32_t loadedSlots = 0U;
    if (slot0 < 0 || slot1 < 0) {
        SetFlag<HardEvent::V_MTE2>(events.vToMte2);
        WaitFlag<HardEvent::V_MTE2>(events.vToMte2);
    }
    if (slot0 < 0) {
        slot0 = slot1 == 0 ? 1 : 0;
        CopyIn(slot0 == 0 ? tensors.input0 : tensors.input1, inputNcOffset + in0 * planeElements + tileOffset, count);
        if (slot0 == 0) {
            cache.cached0 = in0;
            loadedSlots |= 1U;
        } else {
            cache.cached1 = in0;
            loadedSlots |= 2U;
        }
        if (in1 == in0) {
            slot1 = slot0;
        }
    }
    if (slot1 < 0) {
        slot1 = 1 - slot0;
        CopyIn(slot1 == 0 ? tensors.input0 : tensors.input1, inputNcOffset + in1 * planeElements + tileOffset, count);
        if (slot1 == 0) {
            cache.cached0 = in1;
            loadedSlots |= 1U;
        } else {
            cache.cached1 = in1;
            loadedSlots |= 2U;
        }
    }
    if (loadedSlots != 0U) {
        SetFlag<HardEvent::MTE2_V>(events.mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(events.mte2ToV);
        if constexpr (!std::is_same_v<T, float>) {
            if ((loadedSlots & 1U) != 0U) {
                Cast(tensors.cachedFp0, tensors.input0, RoundMode::CAST_NONE, count);
            }
            if ((loadedSlots & 2U) != 0U) {
                Cast(tensors.cachedFp1, tensors.input1, RoundMode::CAST_NONE, count);
            }
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline uint32_t ResizeUpsampleTrilinearSimd<T>::GetOutputsPerBatch(int64_t planeElements, int64_t tileOffset,
                                                                              uint32_t count) const
{
    constexpr uint32_t MAX_OUTPUTS_PER_BATCH = 4;
    constexpr uint32_t MAX_BATCH_COPY_BYTES = 32 * 1024;
    int64_t maxPlaneElements = min(OUTPUT_TILE_ELEMENTS / 2,
                                   static_cast<int64_t>(MAX_BATCH_COPY_BYTES / (2 * sizeof(T))));
    bool fullPlane = planeElements <= maxPlaneElements && tileOffset == 0 &&
                     count == static_cast<uint32_t>(planeElements);
    if (!fullPlane) {
        return 1U;
    }
    uint32_t outputsByUb = static_cast<uint32_t>(OUTPUT_TILE_ELEMENTS / planeElements);
    uint32_t outputsByCopy = static_cast<uint32_t>(MAX_BATCH_COPY_BYTES /
                                                   (planeElements * static_cast<int64_t>(sizeof(T))));
    return min(MAX_OUTPUTS_PER_BATCH, min(outputsByUb, outputsByCopy));
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::ProcessTile(DOnlyLocalTensors<T>& tensors,
                                                                   const DOnlyEvents& events, int64_t inputNcOffset,
                                                                   int64_t outputNcOffset, int64_t planeElements,
                                                                   int64_t tileOffset, uint32_t count)
{
    uint32_t outputsPerBatch = GetOutputsPerBatch(planeElements, tileOffset, count);
    DOnlyCacheState cache;
    int64_t batchStart = 0;
    for (int64_t outIndex = 0; outIndex < tilingData_->outD; ++outIndex) {
        float source = tilingData_->alignCorners == 1 ?
                           static_cast<float>(outIndex) * tilingData_->scaleD :
                           (static_cast<float>(outIndex) + 0.5f) * tilingData_->scaleD - 0.5f;
        float realIndex = source < 0.0f ? 0.0f : source;
        int64_t in0 = min(static_cast<int64_t>(realIndex), tilingData_->inD - 1);
        int64_t in1 = min(in0 + 1, tilingData_->inD - 1);
        float weight1 = min(max(realIndex - static_cast<float>(in0), 0.0f), 1.0f);
        float weight0 = 1.0f - weight1;
        int32_t slot0 = 0, slot1 = 0;
        LoadDepthPair(tensors, cache, events, inputNcOffset, planeElements, tileOffset, count, in0, in1, slot0, slot1);
        uint32_t batchIndex = static_cast<uint32_t>(outIndex - batchStart);
        uint32_t localOffset = batchIndex * count;
        Compute(slot0 == 0 ? tensors.cachedFp0 : tensors.cachedFp1, slot1 == 0 ? tensors.cachedFp0 : tensors.cachedFp1,
                tensors.outputFp, localOffset, count, weight0, weight1);
        CastOutput(tensors.outputRaw, tensors.outputFp, localOffset, count);
        uint32_t batchCount = batchIndex + 1U;
        if (batchCount == outputsPerBatch || outIndex + 1 == tilingData_->outD) {
            CopyOut(tensors.outputRaw, outputNcOffset + batchStart * planeElements + tileOffset, batchCount * count,
                    events.vToMte3, events.mte3ToV);
            batchStart = outIndex + 1;
        }
    }
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::ProcessNc(DOnlyLocalTensors<T>& tensors,
                                                                 const DOnlyEvents& events, int64_t nc,
                                                                 int64_t planeElements, int64_t inputNcElements,
                                                                 int64_t outputNcElements)
{
    for (int64_t tileOffset = 0; tileOffset < planeElements; tileOffset += INPUT_TILE_ELEMENTS) {
        uint32_t count = static_cast<uint32_t>(min(INPUT_TILE_ELEMENTS, planeElements - tileOffset));
        ProcessTile(tensors, events, nc * inputNcElements, nc * outputNcElements, planeElements, tileOffset, count);
    }
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimd<T>::Process()
{
    int64_t totalNc = tilingData_->lenN * tilingData_->lenC;
    int64_t ncStart = 0;
    int64_t ncCount = 0;
    if (!GetNcRange(totalNc, ncStart, ncCount)) {
        return;
    }
    int64_t planeElements = tilingData_->inH * tilingData_->inW;
    int64_t inputNcElements = tilingData_->inD * planeElements;
    int64_t outputNcElements = tilingData_->outD * planeElements;
    DOnlyLocalTensors<T> tensors;
    DOnlyEvents events;
    InitLocalTensors(tensors);
    InitEvents(events);
    for (int64_t nc = ncStart; nc < ncStart + ncCount; ++nc) {
        ProcessNc(tensors, events, nc, planeElements, inputNcElements, outputNcElements);
    }
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMD_H_
