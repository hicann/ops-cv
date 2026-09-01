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
 * \file upsample_bicubic2d_310p.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_310P
#define UPSAMPLE_BICUBIC2D_310P

#include "upsample_bicubic2d_310p_base.h"

namespace UpsampleBicubic2d {

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                        const UpsampleBicubic2dTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    pipe.InitBuffer(centerQueueW, maxDataCount * sizeof(float));          // 2k
    pipe.InitBuffer(xIntQueueW, maxDataCount * sizeof(float));            // 2k
    pipe.InitBuffer(xMinQueueW, maxDataCount * sizeof(float));            // 2k
    pipe.InitBuffer(xVQueueW, maxDataCount * sizeof(float));              // 2k
    pipe.InitBuffer(ratioQueueW, DEFAULT_SLICE_SIZE * 4 * sizeof(float)); // 256

    pipe.InitBuffer(centerQueueH, maxDataCount * sizeof(float));          // 2k
    pipe.InitBuffer(xIntQueueH, maxDataCount * sizeof(float));            // 2k
    pipe.InitBuffer(xMinQueueH, maxDataCount * sizeof(float));            // 2k
    pipe.InitBuffer(xVQueueH, maxDataCount * sizeof(float));              // 2k
    pipe.InitBuffer(ratioQueueH, DEFAULT_SLICE_SIZE * 4 * sizeof(float)); // 256

    pipe.InitBuffer(inputQueue, BUFFER_NUM, maxDataCount * sizeof(float));  // 4k
    pipe.InitBuffer(outputQueue, BUFFER_NUM, maxDataCount * sizeof(float)); // 4k
    pipe.InitBuffer(cacheTensorBuff, maxDataCount * sizeof(float));         // 2k
    pipe.InitBuffer(castInputBuff, maxDataCount * sizeof(float));           // 2k
    pipe.InitBuffer(castOutputBuff, maxDataCount * sizeof(float));          // 2k
    pipe.InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));    // 20k or 40k

    inTensorsGM.SetGlobalBuffer((__gm__ T*)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)output);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::Process()
{
    ClearGM();
    SyncAll();

    BicubicComputeBatch();
    BicubicComputeTail();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::ClearGM()
{
    LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
    int64_t totalNum = outputH * outputW * inputN * inputC;
    int64_t totalBlockNum = (totalNum + blockSize - 1) / blockSize;
    int64_t preCoreBlockCnt = totalBlockNum / needCoreNum;
    int64_t tailBlockCnt = totalBlockNum % needCoreNum;
    int32_t realNeedCore = 1;
    if (preCoreBlockCnt > 0) {
        realNeedCore = needCoreNum;
    }
    if (blockIdx >= realNeedCore) {
        return;
    }
    int64_t preCoreDataCnt = preCoreBlockCnt * blockSize;
    int32_t loopCnt = preCoreDataCnt / DEFAULT_CLEAR_UB_SIZE;
    int64_t tailNum = preCoreDataCnt % DEFAULT_CLEAR_UB_SIZE;
    int64_t offset = blockIdx * preCoreDataCnt;

    Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

    event_t eventIdVToMTE3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    for (int i = 0; i < loopCnt; i++) {
        DataCopy(outTensorsGM[offset], clearUb, DEFAULT_CLEAR_UB_SIZE);
        offset += DEFAULT_CLEAR_UB_SIZE;
    }
    if (tailNum > 0) {
        tailNum = (tailNum + blockSize - 1) / blockSize * blockSize;
        DataCopy(outTensorsGM[offset], clearUb, tailNum);
    }
    if ((tailBlockCnt > 0) && (blockIdx == 0)) {
        tailNum = tailBlockCnt * blockSize;
        offset = preCoreDataCnt * realNeedCore;
        DataCopy(outTensorsGM[offset], clearUb, tailNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::BicubicComputeBatch()
{
    // 计算批量分组的数据
    if (slideStartW >= slideEndW) {
        return;
    }
    slideEndW = Min(slideEndW, outputW);
    int64_t slideOffset = slideEndW - slideStartW;
    int64_t loopCntW = (slideOffset + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    int64_t loopCntH = (outputH + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    for (int64_t loopIdxW = 0; loopIdxW < loopCntW; loopIdxW++) {
        startIdxW = slideStartW + loopIdxW * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t ratioLengthW = Min(slideEndW - startIdxW, DEFAULT_UB_MAX_DATA_COUNT);
        int64_t endIdxW = Min(slideEndW, startIdxW + DEFAULT_UB_MAX_DATA_COUNT);
        CalculateIntermediateTensor(startIdxW, ratioLengthW, W_DIRECTION);
        for (int64_t loopIdxH = 0; loopIdxH < loopCntH; loopIdxH++) {
            startIdxH = loopIdxH * DEFAULT_UB_MAX_DATA_COUNT;
            int64_t ratioLengthH = Min(outputH - startIdxH, DEFAULT_UB_MAX_DATA_COUNT);
            int64_t endIdxH = Min(outputH, startIdxH + DEFAULT_UB_MAX_DATA_COUNT);
            CalculateIntermediateTensor(startIdxH, ratioLengthH, H_DIRECTION);
            for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
                int64_t lenW = Min(slideSize, endIdxW - indexW);
                CalculateRatioTensor(indexW - startIdxW, lenW, W_DIRECTION);
                for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
                    int64_t lengthH = Min(slideSize, endIdxH - indexH);
                    CalculateRatioTensor(indexH - startIdxH, lengthH, H_DIRECTION);
                    CalculateConvolution(indexW, indexH, lenW, lengthH);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::BicubicComputeTail()
{
    // 处理尾块部分数据
    if (tailSlideStartW >= tailSlideEndW) {
        return;
    }
    int64_t slideOffset = tailSlideEndW - tailSlideStartW;
    int64_t loopCntW = (slideOffset + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    int64_t tailRowCnt = tailRowEndW - tailRowStartW;
    int64_t loopCntH = (tailRowCnt + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    for (int64_t loopIdxW = 0; loopIdxW < loopCntW; loopIdxW++) {
        startIdxW = tailSlideStartW + loopIdxW * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t ratioLengthW = Min(DEFAULT_UB_MAX_DATA_COUNT, tailSlideEndW - startIdxW);
        int64_t endIdxW = Min(tailSlideEndW, startIdxW + DEFAULT_UB_MAX_DATA_COUNT);
        CalculateIntermediateTensor(startIdxW, ratioLengthW, W_DIRECTION);
        for (int64_t loopIdxH = 0; loopIdxH < loopCntH; loopIdxH++) {
            startIdxH = tailRowStartW + loopIdxH * DEFAULT_UB_MAX_DATA_COUNT;
            int64_t ratioLengthH = Min(DEFAULT_UB_MAX_DATA_COUNT, tailRowEndW - startIdxH);
            int64_t endIdxH = Min(tailRowEndW, startIdxH + DEFAULT_UB_MAX_DATA_COUNT);
            CalculateIntermediateTensor(startIdxH, ratioLengthH, H_DIRECTION);
            for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
                int64_t lengthW = Min(slideSize, endIdxW - indexW);
                CalculateRatioTensor(indexW - startIdxW, lengthW, W_DIRECTION);
                for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
                    int64_t lengthH = Min(slideSize, endIdxH - indexH);
                    CalculateRatioTensor(indexH - startIdxH, lengthH, H_DIRECTION);
                    CalculateConvolution(indexW, indexH, lengthW, lengthH);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateIntermediateTensor(int64_t index, int64_t length,
                                                                               int8_t direction)
{
    length = Max(length, EACH_SLICEHANDLE_NUM);
    float scale = scaleW;
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    LocalTensor<float> xMinTensor = xMinQueueW.Get<float>();
    LocalTensor<float> xVTensor = xVQueueW.Get<float>();
    if (direction == H_DIRECTION) {
        scale = scaleH;
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        xMinTensor = xMinQueueH.Get<float>();
        xVTensor = xVQueueH.Get<float>();
    }
#if __CCE_AICORE__ == 200
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
#else
    for (int32_t i = 0; i < length; i++) {
        centerTensor.SetValue(i, static_cast<float>(index + i));
    }
#endif

    // 计算center下标
    if (alignCorners) {
        // 角对齐
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 边对齐
        for (int64_t i = 0; i < length; i++) {
            float center = ((float)0.5 + static_cast<float>(index + i)) * scale - (float)0.5;
            centerTensor.SetValue(i, center);
        }
    }

    Floor(xIntTensor, centerTensor, length);
    Adds(xMinTensor, xIntTensor, (float)(-1.0), length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
    Sub(xVTensor, centerTensor, xIntTensor, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateRatioTensor(int64_t xIndex, int64_t length,
                                                                        int8_t direction)
{
    LocalTensor<float> ratioTensor = ratioQueueW.Get<float>();
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    LocalTensor<float> xMinTensor = xMinQueueW.Get<float>();
    LocalTensor<float> xVTensor = xVQueueW.Get<float>();
    int64_t boundSize = inputW;
    if (direction == H_DIRECTION) {
        ratioTensor = ratioQueueH.Get<float>();
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        xMinTensor = xMinQueueH.Get<float>();
        xVTensor = xVQueueH.Get<float>();
        boundSize = inputH;
    }

    // 计算系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());

    int64_t xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    for (int64_t i = 0; i < length; i++) {
        int64_t xSize = 4;
        int64_t idx = i + xIndex;
        if (static_cast<int64_t>(xMinTensor.GetValue(idx)) + 4 > boundSize) {
            xSize = boundSize - static_cast<int64_t>(xMinTensor.GetValue(idx));
        }
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(idx), xIntTensor.GetValue(idx), j, boundSize);
            int64_t weightIndex = j + i * 4;
            ratioTensor.SetValue(weightIndex, w);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateConvolution(int64_t indexW, int64_t indexH, int64_t lengthW,
                                                                        int64_t lengthH)
{
    xMinTensorW = xMinQueueW.Get<float>();
    xMinTensorH = xMinQueueH.Get<float>();
    ratioTensorW = ratioQueueW.Get<float>();
    ratioTensorH = ratioQueueH.Get<float>();
    cacheTensor = cacheTensorBuff.Get<float>();
    castInputTensor = castInputBuff.Get<float>();
    castOutputTensor = castOutputBuff.Get<float>();

    for (int64_t i = 0; i < lengthH; i++) {
        for (int64_t j = 0; j < lengthW; j++) {
            CubicInterp2d(indexW, indexH, j, i);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CopyIn(int64_t indexInput, int64_t calCount)
{
    LocalTensor<T> srcDataLocal = inputQueue.AllocTensor<T>();
    DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount);
    inputQueue.EnQue(srcDataLocal);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CopyOut(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = outputQueue.DeQue<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        int64_t blockCalCnt = (calCount + blockSize - 1) / blockSize * blockSize;
        SetAtomicAdd<T>();
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, blockCalCnt);
        SetAtomicNone();
    }

    outputQueue.FreeTensor(dstDataLocal);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CubicInterp2d(int64_t indexW, int64_t indexH, int64_t offsetW,
                                                                 int64_t offsetH)
{
    int64_t startX = static_cast<int64_t>(xMinTensorW.GetValue(indexW + offsetW - startIdxW));
    int64_t startY = static_cast<int64_t>(xMinTensorH.GetValue(indexH + offsetH - startIdxH));
    int32_t xSize = (inputW - startX) > 4 ? 4 : (inputW - startX);
    int32_t ySize = (inputH - startY) > 4 ? 4 : (inputH - startY);
    int32_t loopCnt = (batchLength + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;

    for (int32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        int64_t startIdx = loopIdx * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t calCount = Min(DEFAULT_UB_MAX_DATA_COUNT, batchLength - startIdx);
        int64_t blockCalCount = (calCount + blockSize - 1) / blockSize * blockSize;

        if (dataType == 2) {
            LocalTensor<float> dstDataLocal = outputQueue.AllocTensor<float>();
            Duplicate(dstDataLocal, (float)0, DEFAULT_UB_MAX_DATA_COUNT);
            for (int32_t y = 0; y < ySize; y++) {
                Duplicate(cacheTensor, (float)0, blockCalCount);
                for (int32_t x = 0; x < xSize; x++) {
                    int64_t indexInput = ((startX + x) + (startY + y) * inputW) * batchLength + startIdx;
                    CopyIn(indexInput, blockCalCount);
                    LocalTensor<float> srcDataLocal = inputQueue.DeQue<float>();
                    float weightW = ratioTensorW.GetValue(offsetW * 4 + x);
                    Muls(srcDataLocal, srcDataLocal, weightW, calCount);
                    Add(cacheTensor, cacheTensor, srcDataLocal, calCount);
                    inputQueue.FreeTensor(srcDataLocal);
                }
                float weightH = ratioTensorH.GetValue(offsetH * 4 + y);
                Muls(cacheTensor, cacheTensor, weightH, calCount);
                Add(dstDataLocal, dstDataLocal, cacheTensor, calCount);
            }
            outputQueue.EnQue(dstDataLocal);
            int64_t indexOutput = ((indexW + offsetW) + (indexH + offsetH) * outputW) * batchLength + startIdx;
            CopyOut(indexOutput, calCount);
        } else {
            LocalTensor<T> dstDataLocal = outputQueue.AllocTensor<T>();
            Duplicate(dstDataLocal, (T)0, DEFAULT_UB_MAX_DATA_COUNT);
            Duplicate(castOutputTensor, (float)0, DEFAULT_UB_MAX_DATA_COUNT);
            for (int32_t y = 0; y < ySize; y++) {
                Duplicate(cacheTensor, (float)0, blockCalCount);
                for (int32_t x = 0; x < xSize; x++) {
                    int64_t indexInput = ((startX + x) + (startY + y) * inputW) * batchLength + startIdx;
                    CopyIn(indexInput, blockCalCount);
                    LocalTensor<T> srcDataLocal = inputQueue.DeQue<T>();
                    float weightW = ratioTensorW.GetValue(offsetW * 4 + x);
                    Cast(castInputTensor, srcDataLocal, RoundMode::CAST_NONE, blockCalCount);
                    Muls(castInputTensor, castInputTensor, weightW, calCount);
                    Add(cacheTensor, cacheTensor, castInputTensor, calCount);
                    inputQueue.FreeTensor(srcDataLocal);
                }
                float weightH = ratioTensorH.GetValue(offsetH * 4 + y);
                Muls(cacheTensor, cacheTensor, weightH, calCount);
                Add(castOutputTensor, castOutputTensor, cacheTensor, calCount);
            }
            Cast(dstDataLocal, castOutputTensor, RoundMode::CAST_NONE, blockCalCount);
            outputQueue.EnQue(dstDataLocal);
            int64_t indexOutput = ((indexW + offsetW) + (indexH + offsetH) * outputW) * batchLength + startIdx;
            CopyOut(indexOutput, calCount);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::ParseTilingData(const UpsampleBicubic2dTilingData* tilingData)
{
    slideSize = DEFAULT_SLICE_SIZE;
    scaleW = tilingData->scale_w;
    scaleH = tilingData->scale_h;
    alignCorners = tilingData->align_corners;
    needCoreNum = tilingData->need_core_num_w;

    inputH = tilingData->input_shapes[0];
    inputW = tilingData->input_shapes[1];
    inputN = tilingData->input_shapes[2];
    inputC = tilingData->input_shapes[3];
    outputH = tilingData->output_shapes[0];
    outputW = tilingData->output_shapes[1];

    batchLength = inputN * inputC;

    slideStartW = tilingData->slideStartList_w[blockIdx];
    slideEndW = tilingData->slideEndList_w[blockIdx];
    tailSlideStartW = tilingData->tailSlideStartList_w[blockIdx];
    tailSlideEndW = tilingData->tailSlideEndList_w[blockIdx];
    tailRowStartW = tilingData->tailRowStartList_w[blockIdx];
    tailRowEndW = tilingData->tailRowEndList_w[blockIdx];

    dataType = tilingData->dataType;

    blockSize = 32 / sizeof(T);
}

} // namespace UpsampleBicubic2d

#endif
