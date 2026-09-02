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
 * \file upsample_nearest_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_NEAREST_BASE_H_
#define _ASCENDC_UPSAMPLE_NEAREST_BASE_H_

#include <type_traits>
#include "kernel_operator.h"

namespace UpsampleNearest {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t NO_BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_MIN_NUM = 16;

// uint8 SmallC: row-pack when a per-column write is smaller than one 32B block.
constexpr int64_t SMALLC_PACK_C = 32;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

const int64_t DEFAULT_UB_MAX_DATA_COUNT = 2048;
const int64_t DEFAULT_UB_MAX_COPY_SIZE = 64 * 1024; // 64kb

template <typename T>
__aicore__ inline constexpr int32_t GatherElemSize()
{
    return sizeof(T) >= sizeof(half) ? sizeof(T) : sizeof(half);
}

template <typename T, int32_t MODE>
class UpsampleNearestND {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearestND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleNearestTilingData* __restrict__ tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };

    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };

    __aicore__ inline void ParseTilingData(const UpsampleNearestTilingData* __restrict__ tilingData);

    __aicore__ inline void CalculateIdxTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void NearestComputeBase();
    __aicore__ inline void NearestComputeSmallCW();
    __aicore__ inline void NearestComputeSmallNCH();
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyInBatch(int64_t indexInput, int64_t calCount, uint16_t blockCnt);
    __aicore__ inline void CopyOutBatch(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyOutBase(LocalTensor<T> dstDataLocal, int64_t indexOutput, int64_t calCount);
    __aicore__ inline void ProcessOutput(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                         int64_t lengthH);
    __aicore__ inline void ProcessOutputBase(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                             int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallC(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                               int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallCW(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                                int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallCWAllBatch(int64_t indexW, int64_t indexH, int64_t lengthW,
                                                        int64_t lengthH);

    __aicore__ inline void GatherWithBridge(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<uint32_t>& offset,
                                            int64_t count, int64_t srcCount = -1);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xIntQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xIntQueueH;
    TBuf<QuePosition::VECCALC> gatherQueue;
    TBuf<QuePosition::VECCALC> offsetQueue;
    TBuf<QuePosition::VECCALC> halfSrcBuf;
    TBuf<QuePosition::VECCALC> halfDstBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    int64_t blockIdx = 0;
    int64_t slideSize = 512;
    float scaleW;
    float scaleH;
    int64_t dataType;

    int64_t tailColStart;
    int64_t tailColEnd;
    int64_t tailRowStart;
    int64_t tailRowEnd;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int32_t blockSize = 8;
    int64_t inputBatchSize;
    int64_t outputBatchSize;
    bool exactMode;
    bool useU16Gather = false;

    int64_t maxCopyCount;
};

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeSmallNCH()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;
    constexpr int32_t bridgeByteSize = GatherElemSize<T>();

    for (int64_t indexH = startIdxH; indexH < endIdxH; indexH++) {
        CalculateIdxTensor(indexH, 1, H_DIRECTION);
        LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(0));
        for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
            int64_t lengthW = Min(slideSize, endIdxW - indexW);
            CalculateIdxTensor(indexW, lengthW, W_DIRECTION);
            LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
            int64_t srcStartW = static_cast<int64_t>(srcTensorW.GetValue(0));

            LocalTensor<int32_t> srcOffsetTensor = offsetQueue.Get<int32_t>();
            Cast(srcOffsetTensor, srcTensorW, RoundMode::CAST_FLOOR, lengthW);
            PipeBarrier<PIPE_V>();
            Adds(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(-srcStartW), lengthW);
            PipeBarrier<PIPE_V>();
            Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(bridgeByteSize), lengthW);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> gatherOffsetTensor = srcOffsetTensor.ReinterpretCast<uint32_t>();

            for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                for (int64_t channelIdx = 0; channelIdx < inputC; channelIdx++) {
                    int64_t indexInput = batchIdx * inputC * inputBatchSize + channelIdx * inputBatchSize +
                                         srcH * inputW + srcStartW;
                    int64_t indexOutput = batchIdx * inputC * outputBatchSize + channelIdx * outputBatchSize +
                                          indexH * outputW + indexW;

                    CopyIn(indexInput, lengthW);

                    LocalTensor<T> srcLocal = dataQueue.DeQue<T>();
                    LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
                    GatherWithBridge(dstDataLocal, srcLocal, gatherOffsetTensor, lengthW);
                    outQueue.EnQue(dstDataLocal);
                    dataQueue.FreeTensor(srcLocal);

                    CopyOutBatch(indexOutput, lengthW);
                }
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeSmallCW()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;
    constexpr int32_t bridgeByteSize = GatherElemSize<T>();

    for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
        int64_t lengthW = Min(slideSize, endIdxW - indexW);
        CalculateIdxTensor(indexW, lengthW, W_DIRECTION);

        LocalTensor<int32_t> gatherTensor = gatherQueue.Get<int32_t>();
        LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
        // B1: ABSOLUTE gather offsets (no minW subtraction) so ProcessOutputSmallCW can load
        // whole source rows from column 0 and address any row via the gather base addr.
        // u16-gather: uint8 even-C gathers channel-pairs as uint16, so the granularity is
        // cPer = inputC/2 elements per output column and the element size is 2 bytes.
        const int64_t cPer = useU16Gather ? (inputC / 2) : inputC;
        const int32_t obytes = useU16Gather ? static_cast<int32_t>(sizeof(uint16_t)) : bridgeByteSize;
        if (inputC == 1) {
            // B2: for C==1 the per-column offset is just srcW; build it with one vector Cast
            // instead of a per-column scalar loop (srcTensorW already holds the source indices).
            Cast(gatherTensor, srcTensorW, RoundMode::CAST_FLOOR, lengthW);
            PipeBarrier<PIPE_V>();
        } else {
            for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
                int32_t srcW = static_cast<int32_t>(srcTensorW.GetValue(offsetW));
                int32_t inputOffset = srcW * static_cast<int32_t>(cPer);
                if (cPer % blockSize == 0) {
                    ArithProgression(gatherTensor[offsetW * cPer], inputOffset, (int32_t)1, cPer);
                } else {
                    for (int64_t i = 0; i < cPer; i++) {
                        gatherTensor.SetValue(offsetW * cPer + i, inputOffset + i);
                    }
                }
            }
        }
        int64_t gatherCount = lengthW * cPer;
        Muls(gatherTensor, gatherTensor, obytes, gatherCount);

        int64_t allInputElems = inputN * inputH * inputW * inputC;
        // Only load the whole input when every input row is actually consumed (upsampling,
        // outputH >= inputH). For downsampling (outputH < inputH) the whole-input load would
        // bring in many unused rows and regress MTE2 -> fall back to per-row loading.
        bool fullLoadOk = (outputH >= inputH) && (allInputElems <= maxCopyCount);
        for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
            int64_t lengthH = Min(slideSize, endIdxH - indexH);
            CalculateIdxTensor(indexH, lengthH, H_DIRECTION);
            if (fullLoadOk) {
                // A: load the whole input (all N batches) in one MTE2, gather every batch's
                // rows from it. Collapses N per-batch loads into 1 (helps multi-batch shapes).
                ProcessOutputSmallCWAllBatch(indexW, indexH, lengthW, lengthH);
            } else {
                for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                    ProcessOutput(batchIdx, indexW, indexH, lengthW, lengthH);
                }
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeBase()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;

    for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
        int64_t lengthH = Min(slideSize, endIdxH - indexH);
        CalculateIdxTensor(indexH, lengthH, H_DIRECTION);
        for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
            int64_t lengthW = Min(slideSize, endIdxW - indexW);
            CalculateIdxTensor(indexW, lengthW, W_DIRECTION);
            for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                ProcessOutput(batchIdx, indexW, indexH, lengthW, lengthH);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CalculateIdxTensor(int64_t index, int64_t length, int8_t direction)
{
    length = Max(length, EACH_SLICE_HANDLE_MIN_NUM);
    float scale = scaleW;
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    float inputSizeBound = static_cast<float>(inputW) - (float)1.0;
    if (direction == H_DIRECTION) {
        scale = scaleH;
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        inputSizeBound = static_cast<float>(inputH) - (float)1.0;
    }

    ArithProgression(centerTensor, static_cast<float>(index), (float)1.0, length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    if (exactMode) {
        // exact模式
        Adds(centerTensor, centerTensor, (float)0.5, length);
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 普通模式
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    }

    Floor(xIntTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xIntTensor, xIntTensor, inputSizeBound, length);
    PipeBarrier<PIPE_V>();
}

} // namespace UpsampleNearest

#endif
