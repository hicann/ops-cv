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
 * \file upsample_nearest.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST
#define UPSAMPLE_NEAREST

#include "upsample_nearest_base.h"

namespace UpsampleNearest {

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                        const UpsampleNearestTilingData* __restrict__ tilingData)
{
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    pipe.InitBuffer(centerQueueW, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float)); // 8k
    pipe.InitBuffer(xIntQueueW, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));   // 8k

    pipe.InitBuffer(centerQueueH, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float)); // 8k
    pipe.InitBuffer(xIntQueueH, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));   // 8k

    maxCopyCount = DEFAULT_UB_MAX_COPY_SIZE / sizeof(T);
    // uint8 SmallC reuses the SmallCW buffer layout so it can gather a whole output row.
    if constexpr (MODE == 1 || (MODE == 2 && std::is_same_v<T, uint8_t>)) {
        constexpr int64_t gatherMinElemSize = GatherElemSize<T>();
        maxCopyCount = DEFAULT_UB_MAX_COPY_SIZE / gatherMinElemSize / 2;
        pipe.InitBuffer(dataQueue, NO_BUFFER_NUM, maxCopyCount * sizeof(T));
        pipe.InitBuffer(outQueue, NO_BUFFER_NUM, maxCopyCount * sizeof(T));
        pipe.InitBuffer(gatherQueue, maxCopyCount * sizeof(uint32_t));
        if constexpr (std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(halfSrcBuf, maxCopyCount * sizeof(half));
            pipe.InitBuffer(halfDstBuf, maxCopyCount * sizeof(half));
        }
    } else if constexpr (MODE == 3) {
        constexpr int64_t gatherMinElemSize = GatherElemSize<T>();
        maxCopyCount = DEFAULT_UB_MAX_COPY_SIZE / gatherMinElemSize / 4;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, maxCopyCount * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, maxCopyCount * sizeof(T));
        pipe.InitBuffer(gatherQueue, maxCopyCount * sizeof(uint32_t));
        pipe.InitBuffer(offsetQueue, maxCopyCount * sizeof(uint32_t));
        if constexpr (std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(halfSrcBuf, maxCopyCount * sizeof(half));
            pipe.InitBuffer(halfDstBuf, maxCopyCount * sizeof(half));
        }
    } else {
        pipe.InitBuffer(dataQueue, BUFFER_NUM, maxCopyCount * sizeof(T)); // 128k
    }

    inTensorsGM.SetGlobalBuffer((__gm__ T*)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)output);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::GatherWithBridge(LocalTensor<T>& dst, LocalTensor<T>& src,
                                                                    LocalTensor<uint32_t>& offset, int64_t count,
                                                                    int64_t srcCount)
{
    if constexpr (std::is_same_v<T, uint8_t>) {
        const int64_t castSrcCount = (srcCount > 0) ? srcCount : count;
        LocalTensor<half> halfSrc = halfSrcBuf.template Get<half>();
        LocalTensor<half> halfDst = halfDstBuf.template Get<half>();
        Cast(halfSrc, src, RoundMode::CAST_NONE, castSrcCount);
        PipeBarrier<PIPE_V>();
        Gather(halfDst, halfSrc, offset, static_cast<uint32_t>(0), count);
        PipeBarrier<PIPE_V>();
        Cast(dst, halfDst, RoundMode::CAST_RINT, count);
    } else {
        (void)srcCount;
        Gather(dst, src, offset, static_cast<uint32_t>(0), count);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::Process()
{
    if (tailColStart >= tailColEnd) {
        return;
    }

    if constexpr (MODE == 1) {
        NearestComputeSmallCW();
    } else if constexpr (MODE == 3) {
        NearestComputeSmallNCH();
    } else {
        NearestComputeBase();
    }
}

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

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutput(int64_t batchIdx, int64_t indexW, int64_t indexH,
                                                                 int64_t lengthW, int64_t lengthH)
{
    if constexpr (MODE == 1) {
        ProcessOutputSmallCW(batchIdx, indexW, indexH, lengthW, lengthH);
    } else if constexpr (MODE == 2) {
        ProcessOutputSmallC(batchIdx, indexW, indexH, lengthW, lengthH);
    } else {
        ProcessOutputBase(batchIdx, indexW, indexH, lengthW, lengthH);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputBase(int64_t batchIdx, int64_t indexW, int64_t indexH,
                                                                     int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
    int32_t loopCnt = (inputC + maxCopyCount - 1) / maxCopyCount;

    for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
        int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;
        int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
        for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
            int64_t srcW = static_cast<int64_t>(srcTensorW.GetValue(offsetW));

            int64_t inputOffset = inputOffsetBase + srcW * inputC;
            int64_t outputOffset = outputOffsetBase + offsetW * inputC;
            for (int32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
                int64_t startIdx = loopIdx * maxCopyCount;
                int64_t calCount = Min(maxCopyCount, inputC - startIdx);

                int64_t indexInput = inputOffset + startIdx;
                CopyIn(indexInput, calCount);
                int64_t indexOutput = outputOffset + startIdx;
                CopyOut(indexOutput, calCount);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputSmallC(int64_t batchIdx, int64_t indexW, int64_t indexH,
                                                                       int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();

    int64_t minW = static_cast<int64_t>(srcTensorW.GetValue(0));
    int64_t maxW = static_cast<int64_t>(srcTensorW.GetValue(lengthW - 1));

    // uint8 sub-block writes (e.g. C=16) bottleneck MTE3. When the source span and one output
    // row both fit UB, gather the whole row and store it in a single aligned MTE3.
    if constexpr (MODE == 2 && std::is_same_v<T, uint8_t>) {
        int64_t spanElems = (maxW - minW + 1) * inputC; // contiguous source columns [minW, maxW]
        int64_t rowOutElems = lengthW * inputC;         // one output row
        if (inputC < SMALLC_PACK_C && spanElems <= maxCopyCount && rowOutElems <= maxCopyCount) {
            constexpr int32_t bridgeByteSize = GatherElemSize<T>();
            LocalTensor<int32_t> gatherTensor = gatherQueue.Get<int32_t>();
            for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
                int32_t base = static_cast<int32_t>((static_cast<int64_t>(srcTensorW.GetValue(offsetW)) - minW) *
                                                    inputC);
                for (int64_t k = 0; k < inputC; k++) {
                    gatherTensor.SetValue(offsetW * inputC + k, base + static_cast<int32_t>(k));
                }
            }
            Muls(gatherTensor, gatherTensor, bridgeByteSize, rowOutElems);
            LocalTensor<uint32_t> gatherOffset = gatherTensor.ReinterpretCast<uint32_t>();
            for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
                int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
                int64_t inBase = (inputBatchSize * batchIdx + srcH * inputW) * inputC + minW * inputC;
                int64_t outBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
                CopyIn(inBase, spanElems);
                LocalTensor<T> src = dataQueue.DeQue<T>();
                LocalTensor<T> dst = outQueue.AllocTensor<T>();
                GatherWithBridge(dst, src, gatherOffset, rowOutElems, spanElems);
                outQueue.EnQue(dst);
                dataQueue.FreeTensor(src);
                CopyOutBatch(outBase, rowOutElems);
            }
            return;
        }
    }

    int64_t inputCBlock = (inputC + blockSize - 1) / blockSize * blockSize;
    int64_t maxDataCopyW = Min(maxCopyCount / inputCBlock, maxW - minW + 1);

    LocalTensor<T> srcDataLocal;
    for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
        int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;

        int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
        int64_t indexInput = inputOffsetBase + minW * inputC;
        int64_t originW = minW + maxDataCopyW;
        CopyInBatch(indexInput, inputC, maxDataCopyW);
        srcDataLocal = dataQueue.DeQue<T>();

        int64_t copyBlockCount = maxDataCopyW;
        for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
            int64_t srcW = static_cast<int64_t>(srcTensorW.GetValue(offsetW));
            int64_t indexOutput = outputOffsetBase + offsetW * inputC;
            if (srcW >= originW) {
                dataQueue.FreeTensor(srcDataLocal);
                indexInput = inputOffsetBase + srcW * inputC;
                if ((copyBlockCount + srcW) > maxW) {
                    copyBlockCount = maxW - srcW + 1;
                }
                CopyInBatch(indexInput, inputC, copyBlockCount);
                originW = srcW + copyBlockCount;
                srcDataLocal = dataQueue.DeQue<T>();
            }
            int64_t indexInputOffset = (srcW + copyBlockCount - originW) * inputCBlock;
            CopyOutBase(srcDataLocal[indexInputOffset], indexOutput, inputC);
        }
        dataQueue.FreeTensor(srcDataLocal);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputSmallCW(int64_t batchIdx, int64_t indexW,
                                                                        int64_t indexH, int64_t lengthW,
                                                                        int64_t lengthH)
{
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
    constexpr int32_t bridgeByteSize = GatherElemSize<T>();

    int64_t maxDataOutEachRow = lengthW * inputC;
    int64_t rowElems = inputW * inputC;         // a full source row (SmallCW: < MAX_SMALL_SHPAE)
    int64_t fullInputElems = inputH * rowElems; // whole input of one batch
    int64_t batchBase = inputBatchSize * batchIdx * inputC;
    LocalTensor<uint32_t> gatherTensor = gatherQueue.Get<uint32_t>();
    const bool u16 = useU16Gather;                  // uint8 && even C: gather channel-pairs as uint16
    const int64_t u16Count = maxDataOutEachRow / 2; // uint16 elements per output row

    if ((outputH >= inputH) && (fullInputElems <= maxCopyCount)) {
        // B1 fast path: one MTE2 load brings in the whole batch input; every output row is
        // gathered from it via the per-row gather base addr. Collapses lengthH tiny loads
        // (the MTE2 bottleneck) into a single transfer. Only when upsampling (outputH>=inputH);
        // for downsampling the whole-batch load would waste MTE2 on unused rows.
        CopyIn(batchBase, fullInputElems);
        LocalTensor<T> srcDataLocal = dataQueue.DeQue<T>();
        LocalTensor<half> halfSrc;
        if constexpr (std::is_same_v<T, uint8_t>) {
            if (!u16) {
                halfSrc = halfSrcBuf.template Get<half>();
                Cast(halfSrc, srcDataLocal, RoundMode::CAST_NONE, fullInputElems); // bridge: cast once
                PipeBarrier<PIPE_V>();
            }
        }
        for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
            int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
            int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
            LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
            if constexpr (std::is_same_v<T, uint8_t>) {
                if (u16) {
                    // u16-gather: no cast, half the elements. Row base in bytes of the uint16
                    // view = srcH * inputW * (inputC/2) * 2 = srcH * rowElems.
                    LocalTensor<uint16_t> src16 = srcDataLocal.template ReinterpretCast<uint16_t>();
                    LocalTensor<uint16_t> dst16 = dstDataLocal.template ReinterpretCast<uint16_t>();
                    Gather(dst16, src16, gatherTensor, static_cast<uint32_t>(srcH * rowElems), u16Count);
                } else {
                    LocalTensor<half> halfDst = halfDstBuf.template Get<half>();
                    uint32_t srcBaseAddr = static_cast<uint32_t>(srcH * rowElems * bridgeByteSize);
                    Gather(halfDst, halfSrc, gatherTensor, srcBaseAddr, maxDataOutEachRow);
                    PipeBarrier<PIPE_V>();
                    Cast(dstDataLocal, halfDst, RoundMode::CAST_RINT, maxDataOutEachRow);
                }
            } else {
                uint32_t srcBaseAddr = static_cast<uint32_t>(srcH * rowElems * bridgeByteSize);
                Gather(dstDataLocal, srcDataLocal, gatherTensor, srcBaseAddr, maxDataOutEachRow);
            }
            outQueue.EnQue(dstDataLocal);
            CopyOutBatch(outputOffsetBase, maxDataOutEachRow);
        }
        dataQueue.FreeTensor(srcDataLocal);
    } else {
        // Fallback (whole input too large for UB): load one full source row per output row.
        // Absolute gather offsets (base 0) keep it correct.
        for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
            int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
            int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;
            int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;

            CopyIn(inputOffsetBase, rowElems);
            LocalTensor<T> srcDataLocal = dataQueue.DeQue<T>();
            LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
            if constexpr (std::is_same_v<T, uint8_t>) {
                if (u16) {
                    LocalTensor<uint16_t> src16 = srcDataLocal.template ReinterpretCast<uint16_t>();
                    LocalTensor<uint16_t> dst16 = dstDataLocal.template ReinterpretCast<uint16_t>();
                    Gather(dst16, src16, gatherTensor, static_cast<uint32_t>(0), u16Count);
                } else {
                    GatherWithBridge(dstDataLocal, srcDataLocal, gatherTensor, maxDataOutEachRow, rowElems);
                }
            } else {
                Gather(dstDataLocal, srcDataLocal, gatherTensor, static_cast<uint32_t>(0), maxDataOutEachRow);
            }
            outQueue.EnQue(dstDataLocal);
            dataQueue.FreeTensor(srcDataLocal);
            CopyOutBatch(outputOffsetBase, maxDataOutEachRow);
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputSmallCWAllBatch(int64_t indexW, int64_t indexH,
                                                                                int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
    constexpr int32_t bridgeByteSize = GatherElemSize<T>();
    int64_t maxDataOutEachRow = lengthW * inputC;
    int64_t rowElems = inputW * inputC;         // one source row
    int64_t fullInputElems = inputH * rowElems; // one batch
    int64_t allInputElems = inputN * fullInputElems;
    LocalTensor<uint32_t> gatherTensor = gatherQueue.Get<uint32_t>();
    const bool u16 = useU16Gather;
    const int64_t u16Count = maxDataOutEachRow / 2;

    // single MTE2 load: the whole input tensor (all N batches), contiguous from offset 0.
    CopyIn(0, allInputElems);
    LocalTensor<T> srcDataLocal = dataQueue.DeQue<T>();
    LocalTensor<half> halfSrc;
    LocalTensor<uint16_t> src16;
    if constexpr (std::is_same_v<T, uint8_t>) {
        if (u16) {
            src16 = srcDataLocal.template ReinterpretCast<uint16_t>();
        } else {
            halfSrc = halfSrcBuf.template Get<half>();
            Cast(halfSrc, srcDataLocal, RoundMode::CAST_NONE, allInputElems); // bridge: cast once
            PipeBarrier<PIPE_V>();
        }
    }
    for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
        int64_t batchElemBase = batchIdx * fullInputElems;
        for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
            int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
            int64_t rowBaseElem = batchElemBase + srcH * rowElems; // element index of this source row
            int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
            LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
            if constexpr (std::is_same_v<T, uint8_t>) {
                if (u16) {
                    LocalTensor<uint16_t> dst16 = dstDataLocal.template ReinterpretCast<uint16_t>();
                    Gather(dst16, src16, gatherTensor, static_cast<uint32_t>(rowBaseElem), u16Count);
                } else {
                    LocalTensor<half> halfDst = halfDstBuf.template Get<half>();
                    Gather(halfDst, halfSrc, gatherTensor, static_cast<uint32_t>(rowBaseElem * bridgeByteSize),
                           maxDataOutEachRow);
                    PipeBarrier<PIPE_V>();
                    Cast(dstDataLocal, halfDst, RoundMode::CAST_RINT, maxDataOutEachRow);
                }
            } else {
                Gather(dstDataLocal, srcDataLocal, gatherTensor, static_cast<uint32_t>(rowBaseElem * bridgeByteSize),
                       maxDataOutEachRow);
            }
            outQueue.EnQue(dstDataLocal);
            CopyOutBatch(outputOffsetBase, maxDataOutEachRow);
        }
    }
    dataQueue.FreeTensor(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyIn(int64_t indexInput, int64_t calCount)
{
    LocalTensor<T> srcDataLocal = dataQueue.AllocTensor<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcDataLocal, inTensorsGM[indexInput], copyParams, padParams);
    }
    dataQueue.EnQue(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOut(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = dataQueue.DeQue<T>();

    CopyOutBase(dstDataLocal, indexOutput, calCount);

    dataQueue.FreeTensor(dstDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOutBase(LocalTensor<T> dstDataLocal, int64_t indexOutput,
                                                               int64_t calCount)
{
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[indexOutput], dstDataLocal, copyParams);
    }
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyInBatch(int64_t indexInput, int64_t calCount, uint16_t blockCnt)
{
    LocalTensor<T> srcDataLocal = dataQueue.AllocTensor<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount * blockCnt);
    } else {
        DataCopyExtParams copyParams{blockCnt, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcDataLocal, inTensorsGM[indexInput], copyParams, padParams);
    }
    dataQueue.EnQue(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOutBatch(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = outQueue.DeQue<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[indexOutput], dstDataLocal, copyParams);
    }
    outQueue.FreeTensor(dstDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ParseTilingData(
    const UpsampleNearestTilingData* __restrict__ tilingData)
{
    slideSize = DEFAULT_UB_MAX_DATA_COUNT;
    dataType = tilingData->dataType;
    scaleW = tilingData->scaleW;
    scaleH = tilingData->scaleH;
    exactMode = tilingData->exactMode;

    inputN = tilingData->inputShapes[0];
    inputH = tilingData->inputShapes[1];
    inputW = tilingData->inputShapes[2];
    inputC = tilingData->inputShapes[3];
    outputH = tilingData->outputShapes[1];
    outputW = tilingData->outputShapes[2];
    if constexpr (MODE == 3) {
        inputN = tilingData->inputShapes[0];
        inputC = tilingData->inputShapes[1];
        inputH = tilingData->inputShapes[2];
        inputW = tilingData->inputShapes[3];
        outputH = tilingData->outputShapes[2];
        outputW = tilingData->outputShapes[3];
    }

    inputBatchSize = inputH * inputW;
    outputBatchSize = outputH * outputW;

    tailColStart = tilingData->tailColStartList[blockIdx];
    tailColEnd = tilingData->tailColEndList[blockIdx];
    tailRowStart = tilingData->tailRowStartList[blockIdx];
    tailRowEnd = tilingData->tailRowEndList[blockIdx];

    blockSize = 32 / sizeof(T);
    // u16-gather fast path: uint8 with even C can gather channel-pairs as uint16,
    // halving the gather element/offset count and removing the half-precision bridge.
    useU16Gather = std::is_same_v<T, uint8_t> && (MODE == 1) && (inputC % 2 == 0);
}

} // namespace UpsampleNearest

#endif
