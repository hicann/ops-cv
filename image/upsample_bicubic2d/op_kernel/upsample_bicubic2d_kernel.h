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
 * \file upsample_bicubic2d_kernel.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D
#define UPSAMPLE_BICUBIC2D

#include "upsample_bicubic2d_kernel_base.h"

namespace UpsampleBicubic2d {

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                    const UpsampleBicubic2dTilingData* tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    int64_t tensorWidthSize = getWidthTensorSize();
    int64_t tensorHeightSize = getHeightTensorSize();

    floatEqual_h = FloatEqual(scale_h, 1.0);
    floatEqual_w = FloatEqual(scale_w, 1.0);

    if (!floatEqual_w) {
        pipe.InitBuffer(centerQueue_w, tensorWidthSize);
        pipe.InitBuffer(xIntQueue_w, tensorWidthSize);
        pipe.InitBuffer(xMinQueue_w, tensorWidthSize);
        pipe.InitBuffer(xVQueue_w, tensorWidthSize);
        pipe.InitBuffer(ratioQueue_w, BUFFER_NUM, ratio_matrix_size_w * sizeof(float));
    }

    if (!floatEqual_h || floatEqual_w) {
        pipe.InitBuffer(centerQueue_h, tensorHeightSize);
        pipe.InitBuffer(xIntQueue_h, tensorHeightSize);
        pipe.InitBuffer(xMinQueue_h, tensorHeightSize);
        pipe.InitBuffer(xVQueue_h, tensorHeightSize);
        pipe.InitBuffer(ratioQueue_h, BUFFER_NUM, ratio_matrix_size_h * sizeof(float));
    }

    intermediateTensorGm.SetGlobalBuffer((__gm__ T*)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T*)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)outTensorsPtr);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    WDirectionExpansion();

    SyncAll();

    // 再纵向扩展
    HDirectionExpansion();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::WDirectionExpansion()
{
    if (!floatEqual_w) {
        if (blockIdx < need_core_num_w) {
            centerTensor = centerQueue_w.Get<float>();
            xIntTensor = xIntQueue_w.Get<float>();
            xMinTensor = xMinQueue_w.Get<float>();
            xVTensor = xVQueue_w.Get<float>();

            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_w < slideEnd_w) {
                for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_w - index);
                    calculateIntermediateTensor(index, length, W_DIRECTION);
                    // 计算系数矩阵
                    calculateRatioTensorW(0, length);
                    copyRatioTensorToGm(0);
                    calculateWidthExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_w < tailSlideEnd_w) {
                calculateIntermediateTensor(tailSlideStart_w, tailSlideEnd_w - tailSlideStart_w, W_DIRECTION);
                for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_w - index);
                    calculateRatioTensorW(0, length);
                    copyRatioTensorToGm(0);
                    calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::HDirectionExpansion()
{
    if (!floatEqual_h || floatEqual_w) {
        if (blockIdx < need_core_num_h) {
            centerTensor = centerQueue_h.Get<float>();
            xIntTensor = xIntQueue_h.Get<float>();
            xMinTensor = xMinQueue_h.Get<float>();
            xVTensor = xVQueue_h.Get<float>();

            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_h < slideEnd_h) {
                for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_h - index);
                    calculateIntermediateTensor(index, length, H_DIRECTION);
                    // 计算系数矩阵
                    calculateRatioTensorH(0, length);
                    copyRatioTensorToGm(1);
                    calculateHeightExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_h < tailSlideEnd_h) {
                calculateIntermediateTensor(tailSlideStart_h, tailSlideEnd_h - tailSlideStart_h, H_DIRECTION);
                for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_h - index);
                    calculateRatioTensorH(0, length);
                    copyRatioTensorToGm(1);
                    calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dND<T>::getWidthTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dND<T>::getHeightTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateIntermediateTensor(int64_t index, int64_t length,
                                                                           int8_t direction)
{
    length = Max(length, EACH_SLICE_HANDLE_NUM);
    float scale = scale_w;
    int64_t max_interp_size = max_interp_size_w;
    if (direction == H_DIRECTION) {
        scale = scale_h;
        max_interp_size = max_interp_size_h;
    }
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    if (align_corners) {
        // 角对齐
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 边对齐
        for (int64_t i = 0; i < length; i++) {
            float center = (static_cast<float>(0.5) + static_cast<float>(index + i)) * scale - static_cast<float>(0.5);
            centerTensor.SetValue(i, center);
        }
        PipeBarrier<PIPE_V>();
    }

    // 计算每个下标的int
    Floor(xIntTensor, centerTensor, length);

    // 计算每个下标的最小映射值
    Adds(xMinTensor, xIntTensor, (float)(-1.0), length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标的v
    Sub(xVTensor, centerTensor, xIntTensor, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateRatioTensorW(int64_t xIndex, int64_t length)
{
    LocalTensor<float> ratioTensor = ratioQueue_w.AllocTensor<float>();
    singleCoreK = 0;
    // 计算横向系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    for (int64_t i = xIndex; i < xIndex + length; i++) {
        int64_t xSize = 4;
        if (static_cast<int64_t>(xMinTensor.GetValue(i)) + 4 > input_shapes[3]) {
            xSize = input_shapes[3] - static_cast<int64_t>(xMinTensor.GetValue(i));
        }
        int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(i), xIntTensor.GetValue(i), j, input_shapes[3]);
            int64_t yIndexValue = j + yIndexOffset;
            singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
            int64_t index = yIndexValue * slide_size + i - xIndex;
            ratioTensor.SetValue(index, w);
        }
    }

    if (dataType != 2) {
        Cast(ratioTensor.ReinterpretCast<T>(), ratioTensor, RoundMode::CAST_RINT, ratioTensor.GetSize());
        ratioQueue_w.EnQue(ratioTensor);
    } else {
        ratioQueue_w.EnQue(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateRatioTensorH(int64_t yIndex, int64_t length)
{
    LocalTensor<float> ratioTensor = ratioQueue_h.AllocTensor<float>();
    xMin = static_cast<int64_t>(xMinTensor.GetValue(yIndex));
    // 计算纵向系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());
    for (int64_t i = yIndex; i < yIndex + length; i++) {
        int64_t xSize = 4;
        if (static_cast<int64_t>(xMinTensor.GetValue(i)) + 4 > input_shapes[2]) {
            xSize = input_shapes[2] - static_cast<int64_t>(xMinTensor.GetValue(i));
        }
        singleCoreK = xMinTensor.GetValue(yIndex + length - 1) - xMin + xSize;
        int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(i), xIntTensor.GetValue(i), j, input_shapes[2]);
            int64_t yIndexValue = j + yIndexOffset;
            int64_t index = yIndexValue + (i - yIndex) * matmulTiling_h->singleCoreK;
            ratioTensor.SetValue(index, w);
        }
    }

    if (dataType != 2) {
        Cast(ratioTensor.ReinterpretCast<T>(), ratioTensor, RoundMode::CAST_RINT, ratioTensor.GetSize());
        ratioQueue_h.EnQue(ratioTensor);
    } else {
        ratioQueue_h.EnQue(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::copyRatioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    if (direction == 0) {
        workSpaceRatioOffset = intermediate_matrix_size + ratio_matrix_size_w * blockIdx;
    } else {
        workSpaceRatioOffset = intermediate_matrix_size + ratio_matrix_size_h * blockIdx;
    }

    if (dataType == 2) {
        LocalTensor<T> ratioTensor = initRatioTensor(direction);
        DataCopy(intermediateTensorGm[workSpaceRatioOffset], ratioTensor, ratioTensor.GetSize());
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRatioTensor(direction, ratioTensor);
    } else {
        int8_t size = 32 / sizeof(T);
        LocalTensor<T> ratioTensor = initRatioTensor(direction);
        DataCopy(intermediateTensorGm[workSpaceRatioOffset], ratioTensor,
                 (ratioTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRatioTensor(direction, ratioTensor);
    }
}

template <typename T>
__aicore__ inline LocalTensor<T> UpsampleBicubic2dND<T>::initRatioTensor(int8_t direction)
{
    if (direction == 0) {
        return ratioQueue_w.DeQue<T>();
    } else {
        return ratioQueue_h.DeQue<T>();
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::releaseRatioTensor(int8_t direction, LocalTensor<T> ratioTensor)
{
    if (direction == 0) {
        return ratioQueue_w.FreeTensor(ratioTensor);
    } else {
        return ratioQueue_h.FreeTensor(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart,
                                                                       int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[3]) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = xMin + rowStart * input_shapes[3];
    int64_t tensorCIdxOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRatioOffset], false);
    if (floatEqual_h) {
        matmulW.IterateAll(outTensorsGM[tensorCIdxOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIdxOffset], false);
    }
    matmulW.End();

    event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID3);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart,
                                                                        int64_t rowEnd)
{
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    if (tensorCIndex + slide_size > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }

    matmulH.SetOrgShape(singleCoreM, output_shapes[3], matmulTiling_h->singleCoreK, output_shapes[2], output_shapes[3]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[2]) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK);
    }
    if (rowEnd == 0) {
        rowEnd = input_shapes[1] * input_shapes[0];
    }

    int64_t xIndex = xMin * output_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex * output_shapes[3];

    int64_t middleHW = input_shapes[2] * output_shapes[3];
    int64_t outputHW = output_shapes[2] * output_shapes[3];

    matmulH.SetTensorA(intermediateTensorGm[workSpaceRatioOffset], false);
    for (int i = rowStart; i < rowEnd; i++) {
        if (floatEqual_w) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * middleHW], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * middleHW], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * outputHW], false);
        matmulH.End();

        event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::ParseTilingData(const UpsampleBicubic2dTilingData* tilingData)
{
    slide_size = tilingData->slide_size;
    scale_w = tilingData->scale_w;
    scale_h = tilingData->scale_h;
    align_corners = tilingData->align_corners;
    max_interp_size_w = tilingData->max_interp_size_w;
    max_interp_size_h = tilingData->max_interp_size_h;

    need_core_num_w = tilingData->need_core_num_w;
    need_core_num_h = tilingData->need_core_num_h;

    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
    }

    ratio_matrix_size_w = (tilingData->ratio_matrix_size_w + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    ratio_matrix_size_h = (tilingData->ratio_matrix_size_h + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    intermediate_matrix_size = tilingData->intermediate_matrix_size / sizeof(T);

    slideStart_w = tilingData->slideStartList_w[blockIdx];
    slideEnd_w = tilingData->slideEndList_w[blockIdx];
    tailSlideStart_w = tilingData->tailSlideStartList_w[blockIdx];
    tailSlideEnd_w = tilingData->tailSlideEndList_w[blockIdx];
    tailRowStart_w = tilingData->tailRowStartList_w[blockIdx];
    tailRowEnd_w = tilingData->tailRowEndList_w[blockIdx];

    slideStart_h = tilingData->slideStartList_h[blockIdx];

    slideEnd_h = tilingData->slideEndList_h[blockIdx];
    tailSlideStart_h = tilingData->tailSlideStartList_h[blockIdx];
    tailSlideEnd_h = tilingData->tailSlideEndList_h[blockIdx];
    tailRowStart_h = tilingData->tailRowStartList_h[blockIdx];
    tailRowEnd_h = tilingData->tailRowEndList_h[blockIdx];

    dataType = tilingData->dataType;

    matmulTiling_h = &tilingData->matmulTiling_h;
    matmulTiling_w = &tilingData->matmulTiling_w;
}

} // namespace UpsampleBicubic2d

#endif
