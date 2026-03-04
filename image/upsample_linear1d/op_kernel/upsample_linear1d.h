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
 * \file upsample_linear1d.h
 * \brief
 */
#ifndef UPSAMPLE_LINEAR1D
#define UPSAMPLE_LINEAR1D

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_linear_common.h"

namespace UpsampleLinear1d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleLinear1dND {
public:
    TPipe pipe;
    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, 
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, MDL_CFG>
        matmulW;

    __aicore__ inline UpsampleLinear1dND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void WDirectionExpansion(int64_t startNum, int64_t endNum, bool isTail);
    __aicore__ inline void RowLoopFunc(int64_t index, int16_t length, int64_t rowOffset, int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, uint64_t inputWorkStartOffset, uint64_t outputWorkStartOffset);
    __aicore__ inline void calculateWidthExtensionFloat(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline void getSlideRange();
    __aicore__ inline void PreLoad(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void AfterMatMul(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void part1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t m_i);
    __aicore__ inline void part3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t index, int16_t length);
    __aicore__ inline void tailPart1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t mmLoopTimes, int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void tailPart3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmLoopTailTimes, int64_t mmLoopTailNum);

private:
    // 系数矩阵下标队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;
    TBuf<TPosition::VECCALC> inputBuf;
    TBuf<TPosition::VECCALC> outputBuf;

    const TCubeTiling *__restrict matmulTiling_w;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<float> intermediateTensorGm;

    bool align_corners = false;
    int64_t blockIdx = 0;
    int64_t slide_size_w = 0;
    float scale_w;

    int64_t need_core_num_w;

    uint32_t radio_matrix_size_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t singleCoreKTiling = 0;
    int64_t xMin = 0;

    int64_t mPerTime;
    int64_t loopTimes;
    int64_t loopTail;
    int64_t inputUbSize;
    int64_t outputUbSize;
    int64_t numPerBlock;
    int64_t matmulLoopTimes;
    int64_t matmulBlockTail;
    int64_t matmulBlockPerTime;

    int64_t loopTailTimes;
    int64_t loopTailTail;
    int64_t remainderMatmulLoopTimes;
    int64_t remainderMatmulBlockTail;
    int64_t remainderLoopTailTimes;
    int64_t remainderLoopTailTail;

    int64_t inputWorkStartOffset_0 = 0;
    int64_t outputWorkStartOffset_0 = 0;
    int64_t inputH = 0;
    int64_t totalPerCore = 0;
};

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    ParseTilingData(tilingData);
    getSlideRange();
    if (!FloatEqual(scale_w, 1.0)) {
        pipe.InitBuffer(inputBuf, inputUbSize);
        pipe.InitBuffer(outputBuf, outputUbSize);
        pipe.InitBuffer(radioQueue, BUFFER_NUM, radio_matrix_size_w * sizeof(float));
    }
    intermediateTensorGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::PreLoad(
    int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow) 
{
    if (numRow <= 0) {
        return ;
    }
    uint32_t numColAlign = (numCol + numPerBlock - 1) / numPerBlock * numPerBlock;
    uint32_t calCount = numRow * numColAlign;

    LocalTensor<float> inputLocalFp32 = inputBuf.Get<float>();
    DataCopyExtParams copyInParams{static_cast<uint16_t>(numRow), static_cast<uint32_t>(numCol * sizeof(T)), static_cast<uint32_t>((input_shapes[3] - numCol) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(numColAlign - numCol), static_cast<T>(0.0)};
    event_t event_v_mte2_0 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    LocalTensor<T> inputLocal = inputLocalFp32.ReinterpretCast<T>()[calCount];
    DataCopyPad(inputLocal, inTensorsGM[inOffset], copyInParams, padParams);
    SetFlag<HardEvent::V_MTE2>(event_v_mte2_0);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte2_0);
    event_t event_mte2_v_0 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event_mte2_v_0);
    WaitFlag<HardEvent::MTE2_V>(event_mte2_v_0);
    Cast(inputLocalFp32, inputLocal, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    event_t event_mte2_mte3_1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_1);
    WaitFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_1);

    event_t event_v_mte3_0 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3_0);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3_0);

    DataCopyExtParams copyOutParams{static_cast<uint16_t>(numRow), static_cast<uint32_t>(numCol * sizeof(float)), static_cast<uint32_t>(((numColAlign - numCol) * sizeof(float)) / 32), 0, 0};
    DataCopyPad(intermediateTensorGm[outOffset], inputLocalFp32, copyOutParams);
    
    event_t event_mte3_mte2_1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_1);
    WaitFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_1);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::AfterMatMul(
    int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow) 
{
    if (numRow <= 0) {
        return ;
    }
    uint32_t calCount = numRow * slide_size_w;
    event_t event_v_mte2_1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event_v_mte2_1);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte2_1);

    LocalTensor<float> outputLocalFp32 = outputBuf.Get<float>();
    LocalTensor<T> outputLocal = outputLocalFp32.ReinterpretCast<T>();
    DataCopy(outputLocalFp32, intermediateTensorGm[inOffset], calCount);
    event_t event_mte2_v_1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event_mte2_v_1);
    WaitFlag<HardEvent::MTE2_V>(event_mte2_v_1);
    if constexpr (std::is_same<T, half>::value) {
        Cast(outputLocal, outputLocalFp32, RoundMode::CAST_NONE, calCount);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(outputLocal, outputLocalFp32, RoundMode::CAST_RINT, calCount);
    } 
    PipeBarrier<PIPE_V>();
    event_t event_mte2_mte3_3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_3);
    WaitFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_3);

    event_t event_v_mte3_1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3_1);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3_1);

    DataCopyExtParams copyOutParams{
        static_cast<uint16_t>(numRow),
        static_cast<uint32_t>(numCol * sizeof(T)),
        static_cast<uint32_t>(((slide_size_w - numCol) * sizeof(T)) / 32),
        static_cast<uint32_t>((output_shapes[3] - numCol) * sizeof(T)),
        0
    };
    DataCopyPad(outTensorsGM[outOffset], outputLocal, copyOutParams);
    event_t event_mte3_mte2_3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_3);
    WaitFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_3);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::Process()
{
    if (FloatEqual(scale_w, 1.0) || blockIdx >= need_core_num_w) {
        return ;
    }
    if constexpr (std::is_same<T, float>::value) {
        if (slideStart_w < slideEnd_w) {
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size_w) {
                int16_t length = Min(slide_size_w, slideEnd_w - index);
                // 计算系数矩阵
                LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
                calculateRadioTensorW(index, length, radioTensor,
                xMin, singleCoreK, scale_w, align_corners, 
                input_shapes[3], slide_size_w, pipe);
                radioQueue.EnQue(radioTensor);
                copyRadioTensorToGm(0);
                calculateWidthExtensionFloat(index, 0, 0);
            }
        }

        // 处理尾块部分数据
        if (tailSlideStart_w < tailSlideEnd_w) {
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size_w) {
                int16_t length = Min(slide_size_w, tailSlideEnd_w - index);
                LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
                calculateRadioTensorW(index, length, radioTensor,
                xMin, singleCoreK, scale_w, align_corners, 
                input_shapes[3], slide_size_w, pipe);
                radioQueue.EnQue(radioTensor);
                copyRadioTensorToGm(0);
                calculateWidthExtensionFloat(index, tailRowStart_w, tailRowEnd_w);
            }
        }
    } else {
        WDirectionExpansion(slideStart_w, slideEnd_w, false);
        WDirectionExpansion(tailSlideStart_w, tailSlideEnd_w, true);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::WDirectionExpansion(int64_t startNum, int64_t endNum, bool isTail)
{
    int64_t rowOffset = isTail ? tailRowStart_w : 0;
    int64_t mmLoopTimes = isTail ? remainderMatmulLoopTimes : matmulLoopTimes;
    int64_t mmBlockTail = isTail ? remainderMatmulBlockTail : matmulBlockTail;
    int64_t mmLoopTailTimes = isTail ? remainderLoopTailTimes : loopTailTimes;
    int64_t mmLoopTailNum = isTail ? remainderLoopTailTail : loopTailTail;
    if (startNum < endNum) {
        for (int64_t index = startNum; index < endNum; index += slide_size_w) {
            int16_t length = Min(slide_size_w, endNum - index);
            // 计算系数矩阵
            LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
            calculateRadioTensorW(index, length, radioTensor,
            xMin, singleCoreK, scale_w, align_corners, 
            input_shapes[3], slide_size_w, pipe);
            radioQueue.EnQue(radioTensor);
            copyRadioTensorToGm(0);
            RowLoopFunc(index, length, rowOffset, mmLoopTimes, mmBlockTail, mmLoopTailTimes, mmLoopTailNum);
        }
    } else {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::calculateWidthExtensionFloat(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    if (singleCoreK <= 0) {
        return ;
    }
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size_w > output_shapes[3]) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = xMin + rowStart * input_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    matmulW.End();

    event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID3);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::part1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t m_i)
{
    int64_t preOutOffset = inputWorkStartOffset;
    int64_t rowNum = 0;
    for (int64_t ii = 0; ii < loopTimes; ii ++) {
        int64_t rowStart = Min(rowOffset + m_i * matmulBlockPerTime + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t preInOffset = rowStart * input_shapes[3] + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }

    if(loopTail > 0) {
        int64_t rowStart = Min(rowOffset + m_i * matmulBlockPerTime + loopTimes * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + loopTail, inputH);
        int64_t preInOffset = rowStart * input_shapes[3] + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::part3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t index, int16_t length)
{
    int64_t afterInOffset = outputWorkStartOffset;
    int64_t rowNum = 0;
    for (int64_t ii = 0; ii < loopTimes; ii ++) {
        int64_t rowStart = Min(rowOffset + m_i * matmulBlockPerTime + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t afterOutOffset = rowStart * output_shapes[3] + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }

    if(loopTail > 0) {
        int64_t rowStart = Min(rowOffset + m_i * matmulBlockPerTime + loopTimes * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + loopTail, inputH);
        int64_t afterOutOffset = rowStart * output_shapes[3] + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::tailPart1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t mmLoopTimes, int64_t mmLoopTailTimes, int64_t mmLoopTailNum)
{
    int64_t preOutOffset = inputWorkStartOffset;
    int64_t rowNum = 0;
    for (int64_t ii = 0; ii < mmLoopTailTimes; ii ++) {
        int64_t rowStart = Min(rowOffset + mmLoopTimes * matmulBlockPerTime + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t preInOffset = rowStart * input_shapes[3] + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }

    if(mmLoopTailNum > 0) {
        int64_t rowStart = Min(rowOffset + mmLoopTimes * matmulBlockPerTime + mmLoopTailTimes * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mmLoopTailNum, inputH);
        int64_t preInOffset = rowStart * input_shapes[3] + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::tailPart3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmLoopTailTimes, int64_t mmLoopTailNum)
{
    int64_t afterInOffset = outputWorkStartOffset;
    int64_t rowNum = 0;
    for (int64_t ii = 0; ii < mmLoopTailTimes; ii ++) {
        int64_t rowStart = Min(rowOffset + mmLoopTimes * matmulBlockPerTime + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t afterOutOffset = rowStart * output_shapes[3] + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }

    if(mmLoopTailNum > 0) {
        int64_t rowStart = Min(rowOffset + mmLoopTimes * matmulBlockPerTime + mmLoopTailTimes * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mmLoopTailNum, inputH);
        int64_t afterOutOffset = rowStart * output_shapes[3] + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::RowLoopFunc(int64_t index, int16_t length, int64_t rowOffset, int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, int64_t mmLoopTailNum)
{
    uint64_t inputWorkStartOffset = inputWorkStartOffset_0;
    uint64_t outputWorkStartOffset = outputWorkStartOffset_0;
    for (int64_t m_i = 0; m_i < mmLoopTimes; m_i ++) {
        if ASCEND_IS_AIV {
            part1(inputWorkStartOffset, rowOffset, m_i);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }

        uint64_t matmulRowStart = m_i * matmulBlockPerTime;
        uint64_t matmulRowEnd = matmulRowStart + matmulBlockPerTime;
        calculateWidthExtension(index, matmulRowStart, matmulRowEnd, inputWorkStartOffset, outputWorkStartOffset);

        if ASCEND_IS_AIV {
            part3(outputWorkStartOffset, rowOffset, m_i, index, length);
        }
    }

    if (mmBlockTail > 0) {
        if ASCEND_IS_AIV {
            tailPart1(inputWorkStartOffset, rowOffset, mmLoopTimes, mmLoopTailTimes, mmLoopTailNum);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        }

        uint64_t matmulRowStart = mmLoopTimes * matmulBlockPerTime;
        uint64_t matmulRowEnd = matmulRowStart + matmulBlockPerTime;
        calculateWidthExtension(index, matmulRowStart, matmulRowEnd, inputWorkStartOffset, outputWorkStartOffset);

        if ASCEND_IS_AIV {
            tailPart3(outputWorkStartOffset, rowOffset, index, length, mmLoopTimes, mmLoopTailTimes, mmLoopTailNum);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::copyRadioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    workSpaceRadioOffset = totalPerCore * blockIdx;
    LocalTensor<float> radioTensor = radioQueue.DeQue<float>();
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, radioTensor.GetSize());
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    radioQueue.FreeTensor(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, uint64_t inputWorkStartOffset, uint64_t outputWorkStartOffset)
{
    if (singleCoreK > 0) {
        int64_t totalM = matmulTiling_w->singleCoreM;
        int64_t singleCoreN = matmulTiling_w->singleCoreN;
        // 尾块batch分批处理
        int64_t singleCoreM = rowEnd - rowStart;
        matmulW.SetOrgShape(singleCoreM, singleCoreN, singleCoreK, singleCoreK, singleCoreN);
        matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
        matmulW.SetTensorA(intermediateTensorGm[inputWorkStartOffset], false);
        matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
        matmulW.IterateAll(intermediateTensorGm[outputWorkStartOffset], false);
        matmulW.End();

        event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::ParseTilingData(const UpsampleLinear1dTilingData *tilingData)
{
    align_corners = tilingData->align_corners;
    slide_size_w = tilingData->slide_size_w;
    scale_w = tilingData->scale_w;

    need_core_num_w = tilingData->need_core_num_w;

    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
        input_shapes[i] = tilingData->input_shapes[i];
    }

    radio_matrix_size_w = (tilingData->radio_matrix_size_w + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    eachCoreSlideNumW = tilingData->eachCoreSlideNumW;
    tailStartSlideNumW = tilingData->tailStartSlideNumW;
    slideNumW = tilingData->slideNumW;
    groupCoreNumW = tilingData->groupCoreNumW;
    tailAvergingRowsW = tilingData->tailAvergingRowsW;
    remainderW = tilingData->remainderW;

    mPerTime = tilingData->mPerTime;
    loopTimes = tilingData->loopTimes;
    loopTail = tilingData->loopTail;
    numPerBlock = 32 / sizeof(T);
    loopTailTimes = tilingData->loopTailTimes;
    loopTailTail = tilingData->loopTailTail;
    remainderLoopTailTimes = tilingData->remainderLoopTailTimes;
    remainderLoopTailTail = tilingData->remainderLoopTailTail;
    inputUbSize = tilingData->inputUbSize;
    outputUbSize = tilingData->outputUbSize;
    matmulLoopTimes = tilingData->matmulLoopTimes;
    matmulBlockTail = tilingData->matmulBlockTail;
    remainderMatmulLoopTimes = tilingData->remainderMatmulLoopTimes;
    remainderMatmulBlockTail = tilingData->remainderMatmulBlockTail;
    matmulBlockPerTime = tilingData->matmulBlockPerTime;
    singleCoreKTiling = tilingData->singleCoreK;

    getWorkSize(totalPerCore, inputWorkStartOffset_0, outputWorkStartOffset_0,
    matmulBlockPerTime, singleCoreKTiling, slide_size_w, radio_matrix_size_w, blockIdx);

    matmulTiling_w = &tilingData->matmulTiling_w;
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::getSlideRange()
{
    inputH = input_shapes[0] * input_shapes[1] * input_shapes[2];
    slideStart_w = blockIdx * eachCoreSlideNumW * slide_size_w;
    slideEnd_w = Min((Min((blockIdx + 1) * eachCoreSlideNumW, slideNumW)) * slide_size_w, output_shapes[3]);
    int64_t groupIndex = groupCoreNumW > 0 ? blockIdx / groupCoreNumW : 0;
    if (groupIndex < remainderW) {
        tailSlideStart_w = (tailStartSlideNumW + groupIndex) * slide_size_w;
        tailSlideEnd_w = Min(tailSlideStart_w + slide_size_w, output_shapes[3]);
        int64_t blockIdxInGroup = groupCoreNumW > 0 ? blockIdx % groupCoreNumW : 0;
        tailRowStart_w = blockIdxInGroup * tailAvergingRowsW;
        tailRowEnd_w = Min(tailRowStart_w + tailAvergingRowsW, inputH);
    }
}

}  // namespace UpsampleLinear1d

#endif  // UPSAMPLE_LINEAR1D