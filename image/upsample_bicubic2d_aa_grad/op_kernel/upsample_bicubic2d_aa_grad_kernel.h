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
 * \file upsample_bicubic2d_aa_grad_kernel.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC_AA_GRAD
#define UPSAMPLE_BICUBIC_AA_GRAD

#include "upsample_bicubic2d_aa_grad_kernel_base.h"

namespace UpSampleBicubic2dAAGrad {

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::getQueueSize()
{
    zeroScaleW = input_shapes[3] > 0 ? static_cast<float>(output_shapes[3]) / input_shapes[3] : 1;
    zeroScaleH = input_shapes[2] > 0 ? static_cast<float>(output_shapes[2]) / input_shapes[2] : 1;

    queueSizeW = scale_w > 0 ? static_cast<int64_t>(2 * (slide_size + support_w) / scale_w) + 1 :
                               static_cast<int64_t>(2 * (slide_size + support_w) / zeroScaleW) + 1;
    queueSizeH = scale_h > 0 ? static_cast<int64_t>(2 * (slide_size + support_h) / scale_h) + 1 :
                               static_cast<int64_t>(2 * (slide_size + support_h) / zeroScaleH) + 1;
    queueSizeW++;
    queueSizeH++;

    queueSize = getMax(static_cast<int64_t>(1), getMax(queueSizeW, queueSizeH));
};

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::WDirectionExpansion()
{
    if (blockIdx < need_core_num_w) {
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> xMinTensor = xMinQueue.Get<float>();
        LocalTensor<float> xSizeTensor = xSizeQueue.Get<float>();
        LocalTensor<float> weightTensor = weightQueue.Get<float>();

        // 计算滑块映射范围
        if (slideStart_w < slideEnd_w) {
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_w - index);
                slidelen = length;
                calculateIntermediateTensorX(centerTensor, xMinTensor, xSizeTensor, weightTensor, index, slideEnd_w);
                calculateRadioTensor(centerTensor, xMinTensor, xSizeTensor, weightTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, 0, 0);
            }
        }
        if (tailSlideStart_w < tailSlideEnd_w) {
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_w - index);
                slidelen = length;
                calculateIntermediateTensorX(centerTensor, xMinTensor, xSizeTensor, weightTensor, index,
                                             tailSlideEnd_w);
                calculateRadioTensor(centerTensor, xMinTensor, xSizeTensor, weightTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
            }
        }
        // 处理尾块部分数据
        centerQueue.FreeTensor(centerTensor);
        xMinQueue.FreeTensor(xMinTensor);
        xSizeQueue.FreeTensor(xSizeTensor);
        weightQueue.FreeTensor(weightTensor);
    }
    // 获取要计算系数矩阵的下标
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::HDirectionExpansion()
{
    if (blockIdx < need_core_num_h) {
        instartIndex = 0;
        inendIndex = 0;
        LocalTensor<float> centerTensor_h = centerQueue.Get<float>();
        LocalTensor<float> xMinTensor_h = xMinQueue.Get<float>();
        LocalTensor<float> xSizeTensor_h = xSizeQueue.Get<float>();
        LocalTensor<float> weightTensor_h = weightQueue.Get<float>();
        if (slideStart_h < slideEnd_h) {
            for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_h + 1 - index);
                slidelen_h = length;
                calculateIntermediateTensorY(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index,
                                             slideEnd_h);
                calculateRadioTensorH(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, 0, 0);
            }
        }

        if (tailSlideStart_h < tailSlideEnd_h) {
            for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_h + 1 - index);
                slidelen_h = length;
                calculateIntermediateTensorY(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index,
                                             tailSlideEnd_h);
                calculateRadioTensorH(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
            }
        }

        // 释放临时tensor
        centerQueue.FreeTensor(centerTensor_h);
        xMinQueue.FreeTensor(xMinTensor_h);
        xSizeQueue.FreeTensor(xSizeTensor_h);
        weightQueue.FreeTensor(weightTensor_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                          UpsampleBicubicAAGradTilingData* tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);

    needExpendX = !FloatEqual(scale_w, 1.0);
    needExpendY = !FloatEqual(scale_h, 1.0);

    getQueueSize();
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    int64_t interpsize = getMax(max_interp_size_h, max_interp_size_w);

    pipe.InitBuffer(centerQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xMinQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xSizeQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(floorQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, (radioSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(weightQueue, (interpsize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioCastQueue, NO_BUFFER_NUM, (radioSize * sizeof(T) + 31) / 32 * 32);

    intermediateTensorGm.SetGlobalBuffer((__gm__ T*)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T*)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)outTensorsPtr);
};

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    if (needExpendX) {
        WDirectionExpansion();
    }

    SyncAll();

    // 再纵向扩展
    if (needExpendY || !needExpendX) {
        HDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateIntermediateTensorX(
    LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor,
    LocalTensor<float> weightTensor, int64_t slideStart_w, int64_t slideEnd_w)
{
    instart_w = scale_w > 0 ? static_cast<int64_t>((float)(slideStart_w - support_w) / scale_w) - 1 :
                              static_cast<int64_t>((float)(slideStart_w - support_w) / zeroScaleW) - 1;

    if (instart_w < 0) {
        instart_w = 0;
    }
    LocalTensor<float> floorTensor = floorQueue.Get<float>();

    int64_t length = queueSizeW;
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor, static_cast<float>(instart_w), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    // 计算center下标
    Adds(centerTensor, centerTensor, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor, centerTensor, scale_w, length);
    PipeBarrier<PIPE_V>();
    // 计算每个下标最小映射值
    Adds(floorTensor, centerTensor, (float)0.5 - support_w, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor, floorTensor, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
    // 计算每个下标映射的范围
    Adds(floorTensor, centerTensor, (float)0.5 + support_w, length);
    PipeBarrier<PIPE_V>();
    Floor(xSizeTensor, floorTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(output_shapes[3]), length);
    PipeBarrier<PIPE_V>();
    Sub(xSizeTensor, xSizeTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(max_interp_size_w), length);
    PipeBarrier<PIPE_V>();
    Maxs(xSizeTensor, xSizeTensor, (float)0.0, length);
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateIntermediateTensorY(
    LocalTensor<float> centerTensor_h, LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h,
    LocalTensor<float> weightTensor_h, int64_t slideStart_h, int64_t slideEnd_h)
{
    instart_h = scale_h > 0 ? static_cast<int64_t>((float)(slideStart_h - support_h) / scale_h) - 1 :
                              static_cast<int64_t>((float)(slideStart_h - support_h) / zeroScaleH) - 1;
    int64_t length = queueSizeH;
    if (instart_h < 0) {
        instart_h = 0;
    }
    LocalTensor<float> floorTensor_h = floorQueue.Get<float>();
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor_h, static_cast<float>(instart_h), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    // 计算center下标
    Adds(centerTensor_h, centerTensor_h, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor_h, centerTensor_h, scale_h, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标最小映射值
    Adds(floorTensor_h, centerTensor_h, (float)0.5 - support_h, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor_h, floorTensor_h, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor_h, xMinTensor_h, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标映射的范围
    Adds(floorTensor_h, centerTensor_h, (float)0.5 + support_h, length);
    PipeBarrier<PIPE_V>();

    Floor(xSizeTensor_h, floorTensor_h, length);
    PipeBarrier<PIPE_V>();

    Mins(xSizeTensor_h, xSizeTensor_h, static_cast<float>(output_shapes[2]), length);
    PipeBarrier<PIPE_V>();

    Sub(xSizeTensor_h, xSizeTensor_h, xMinTensor_h, length);
    PipeBarrier<PIPE_V>();

    Mins(xSizeTensor_h, xSizeTensor_h, static_cast<float>(max_interp_size_h), length);
    PipeBarrier<PIPE_V>();

    Maxs(xSizeTensor_h, xSizeTensor_h, (float)0.0, length);
    // 计算批量分组的数据
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::computeIndexValueH(LocalTensor<float> xMinTensor_h,
                                                                        LocalTensor<float> xSizeTensor_h, int64_t index,
                                                                        int64_t length)
{
    instartIndex = 0;
    inendIndex = 0;
    for (; instartIndex < queueSizeH; instartIndex++) {
        int64_t ymax = xMinTensor_h.GetValue(instartIndex) + xSizeTensor_h.GetValue(instartIndex);
        if (ymax >= index) {
            break;
        }
    }
    for (inendIndex = instartIndex; inendIndex < queueSizeH; inendIndex++) {
        if (xMinTensor_h.GetValue(inendIndex) > index + length - 1) {
            break;
        } else if (inendIndex + instart_h > input_shapes[2] - 1) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateRadioTensorH(LocalTensor<float> centerTensor_h,
                                                                           LocalTensor<float> xMinTensor_h,
                                                                           LocalTensor<float> xSizeTensor_h,
                                                                           LocalTensor<float> weightTensor_h,
                                                                           int64_t index, int64_t length)
{
    LocalTensor<float> radioTensor_h = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor_h, float(0.0), radioTensor_h.GetSize());

    // 计算影响该块的原始矩阵点的下标

    computeIndexValueH(xMinTensor_h, xSizeTensor_h, index, length);
    singleCoreK_h = inendIndex - instartIndex;
    for (int64_t i = instartIndex; i < inendIndex; i++) {
        float total_w = 0.0;
        int64_t xmin = xMinTensor_h.GetValue(i);
        int64_t xmax = xmin + xSizeTensor_h.GetValue(i);
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor_h.GetValue(i)); j++) {
            float w = getWeight((j + xMinTensor_h.GetValue(i) - centerTensor_h.GetValue(i) + (float)0.5) * invscale_h);
            weightTensor_h.SetValue(j, w);
            total_w += w;
        }
        int64_t insertx = i - instartIndex;
        singleCoreK_h = singleCoreK_h < insertx + 1 ? insertx + 1 : singleCoreK_h;
        int64_t xstart = getMax(index, xmin) - index;
        int64_t xend = getMin(index + slidelen_h, xmax) - index;
        if (!FloatEqual(total_w, 0.0)) {
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor_h.GetValue(i)); j++) {
                float weight = weightTensor_h.GetValue(j) / total_w;
                // 求更新系数矩阵中行的位置

                int64_t yIndexValue = xmin + j - index;

                if (yIndexValue < xend && yIndexValue >= 0) {
                    int64_t index = yIndexValue * matmulTiling_h->singleCoreK + insertx;
                    radioTensor_h.SetValue(index, weight);
                }
            }
        }
    }

    if (dataType != 2) {
        LocalTensor<T> radioCastTensor_h = radioCastQueue.AllocTensor<T>();
        Cast(radioCastTensor_h, radioTensor_h, RoundMode::CAST_RINT, radioTensor_h.GetSize());
        radioCastQueue.EnQue(radioCastTensor_h);
        radioQueue.FreeTensor(radioTensor_h);
    } else {
        radioQueue.EnQue(radioTensor_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::computeIndexValueW(LocalTensor<float> xMinTensor,
                                                                        LocalTensor<float> xSizeTensor, int64_t index,
                                                                        int64_t length)
{
    instartIndex = 0;
    inendIndex = 0;
    for (; instartIndex < queueSizeW; instartIndex++) {
        int64_t xmax = xMinTensor.GetValue(instartIndex) + xSizeTensor.GetValue(instartIndex);
        if (xmax >= index) {
            break;
        }
    }

    for (inendIndex = instartIndex; inendIndex < queueSizeW; inendIndex++) {
        if (xMinTensor.GetValue(inendIndex) > index + length - 1) {
            break;
        } else if (inendIndex + instart_w > input_shapes[3] - 1) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateRadioTensor(LocalTensor<float> centerTensor,
                                                                          LocalTensor<float> xMinTensor,
                                                                          LocalTensor<float> xSizeTensor,
                                                                          LocalTensor<float> weightTensor,
                                                                          int64_t index, int64_t length)
{
    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
    // 计算影响该块的原始矩阵点的下标
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    computeIndexValueW(xMinTensor, xSizeTensor, index, length);

    for (int64_t i = instartIndex; i < inendIndex; i++) {
        float total_w = 0.0;
        int64_t xmin = xMinTensor.GetValue(i);
        int64_t xmax = xmin + xSizeTensor.GetValue(i);

        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
            float w = getWeight((j + xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5) * invscale_w);

            weightTensor.SetValue(j, w);
            total_w += w;
        }

        if (!FloatEqual(total_w, 0.0)) {
            int64_t xstart = getMax(index, xmin) - index;
            int64_t xend = getMin(index + length, xmax) - index;
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / total_w;
                // 求更新系数矩阵中行的位置
                int64_t insertx = xmin + j - index;

                if (insertx < xend && insertx >= 0) {
                    int64_t yIndexValue = 0;

                    yIndexValue = i - instartIndex;

                    singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
                    if (instart_w + instartIndex + singleCoreK > input_shapes[3]) {
                        singleCoreK = input_shapes[3] - instartIndex - instart_w;
                    }
                    int64_t index = yIndexValue * length + insertx;

                    radioTensor.SetValue(index, weight);
                }
            }
        }
    }

    if (dataType != 2) {
        LocalTensor<T> radioCastLocalTensorW = radioCastQueue.AllocTensor<T>();
        Cast(radioCastLocalTensorW, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
        radioCastQueue.EnQue(radioCastLocalTensorW);
        radioQueue.FreeTensor(radioTensor);
    } else {
        radioQueue.EnQue(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::copyRadioTensorToGm()
{
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    workSpaceRadioOffset = intermediate_matrix_size + radioSize * blockIdx;
    int8_t size = 32 / sizeof(T);

    if (dataType == 2) {
        LocalTensor<T> radioBuf = radioQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioBuf, (radioBuf.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        radioQueue.FreeTensor(radioBuf);
    } else {
        LocalTensor<T> radioCastTensor = radioCastQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioCastTensor,
                 (radioCastTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        radioCastQueue.FreeTensor(radioCastTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart,
                                                                             int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    if (singleCoreK == 0) {
        singleCoreK++;
    }

    if (tensorCIndex + slide_size > output_shapes[3]) {
        singleCoreN = slidelen;
    }

    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);

    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[3] - 1) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = instartIndex + instart_w + rowStart * input_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);

    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);

    if (!needExpendY) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart,
                                                                              int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    if (singleCoreK_h == 0) {
        singleCoreK_h++;
    }
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreN = rowEnd - rowStart;
    }

    if (tensorCIndex + slide_size > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }
    matmulH.SetOrgShape(singleCoreM, output_shapes[3], matmulTiling_h->singleCoreK, output_shapes[2], output_shapes[3]);

    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK_h);

    if (tensorCIndex + slide_size > output_shapes[2] - 1) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK_h);
    }

    int64_t xIndex = (instartIndex + instart_h) * output_shapes[3] + rowStart;

    int64_t tensorCIndexWithOffset = tensorCIndex * output_shapes[3] + rowStart;

    for (int i = 0; i < output_shapes[0] * output_shapes[1]; i++) {
        // 系数矩阵起始位置
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needExpendX) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * input_shapes[2] * output_shapes[3]], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * input_shapes[2] * output_shapes[3]], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * output_shapes[2] * output_shapes[3]], false);
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::ParseTilingData(UpsampleBicubicAAGradTilingData* tilingData)
{
    slide_size = tilingData->slide_size;
    scale_w = tilingData->scale_w;
    scale_h = tilingData->scale_h;
    invscale_w = tilingData->invscale_w;
    invscale_h = tilingData->invscale_h;

    support_h = tilingData->support_h;
    support_w = tilingData->support_w;
    max_interp_size_h = tilingData->max_interp_size_h;
    max_interp_size_w = tilingData->max_interp_size_w;

    need_core_num_h = tilingData->need_core_num_h;
    need_core_num_w = tilingData->need_core_num_w;

    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
    }

    intermediate_matrix_size = tilingData->intermediate_matrix_size;
    radio_matrix_size = tilingData->radio_matrix_size;
    radio_matrix_size_h = tilingData->radio_matrix_size_h;

    slideStart_w = tilingData->slideStartList_w[blockIdx];
    tailSlideStart_w = tilingData->tailSlideStartList_w[blockIdx];
    tailRowStart_w = tilingData->tailRowStartList_w[blockIdx];
    slideEnd_w = tilingData->slideEndList_w[blockIdx];
    tailSlideEnd_w = tilingData->tailSlideEndList_w[blockIdx];
    tailRowEnd_w = tilingData->tailRowEndList_w[blockIdx];

    slideStart_h = tilingData->slideStartList_h[blockIdx];
    tailSlideStart_h = tilingData->tailSlideStartList_h[blockIdx];
    tailRowStart_h = tilingData->tailRowStartList_h[blockIdx];
    slideEnd_h = tilingData->slideEndList_h[blockIdx];
    tailSlideEnd_h = tilingData->tailSlideEndList_h[blockIdx];
    tailRowEnd_h = tilingData->tailRowEndList_h[blockIdx];

    matmulTiling_h = &tilingData->matmulTiling_h;
    matmulTiling_w = &tilingData->matmulTiling_w;
    dataType = tilingData->dataType;
}

} // namespace UpSampleBicubic2dAAGrad

#endif
