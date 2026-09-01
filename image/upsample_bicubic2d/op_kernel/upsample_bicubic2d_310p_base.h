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
 * \file upsample_bicubic2d_310p_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BICUBIC2D_310P_BASE_H_
#define _ASCENDC_UPSAMPLE_BICUBIC2D_310P_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2d {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t EACH_SLICEHANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr int8_t MIN_SIZE = 1;
constexpr int8_t TWO_SIZE = 2;

const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
const int32_t DEFAULT_SLICE_SIZE = 16;
const int32_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;
const int64_t DEFAULT_UB_MAX_DATA_COUNT = 512;

template <typename T>
class UpsampleBicubic2dND310p {
public:
    TPipe pipe;

    __aicore__ inline UpsampleBicubic2dND310p(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleBicubic2dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 x, int64_t i, int64_t j, int64_t width)
    {
        float weight1 = 0;
        float weight2 = 0;
        float weight3 = 0;
        float weight4 = 0;
        float t = (float)1.0 - x;
        switch (j) {
            case 0:
                weight1 = calWeights2(1 + x);
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                return getWeightIndex0(i, width, weight1, weight2, weight3);
            case 1:
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex1(i, width, weight2, weight3, weight4);
            case 2:
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex2(i, width, weight3, weight4);
            case 3:
                weight4 = calWeights2(1 + t);
                return getWeightIndex3(i, width, weight4);
            default:
                return 0.0;
        }
    };

    template <typename T1>
    __aicore__ inline T1 getWeightIndex0(int64_t i, int64_t width, T1 weight1, T1 weight2, T1 weight3)
    {
        if (width == MIN_SIZE) {
            return 1.0;
        } else if (i < 0) {
            return (weight1 + weight2 + weight3);
        } else if (i == 0) {
            return (weight1 + weight2);
        } else if (out_of_range(i, width)) {
            return weight1;
        } else if (on_board(i, width)) {
            return weight1;
        } else {
            return weight1;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex1(int64_t i, int64_t width, T1 weight2, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE) {
            return 0.0;
        } else if (i < 0) {
            return weight4;
        } else if (i == 0) {
            return (width == TWO_SIZE) ? (weight3 + weight4) : weight3;
        } else if (out_of_range(i, width)) {
            return (weight2 + weight3 + weight4);
        } else if (on_board(i, width)) {
            return weight2;
        } else {
            return weight2;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex2(int64_t i, int64_t width, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE || i < 0) {
            return 0.0;
        } else if (i == 0) {
            return (width == TWO_SIZE) ? static_cast<float>(0.0) : weight4;
        } else if (out_of_range(i, width)) {
            return 0.0;
        } else if (on_board(i, width)) {
            return (weight3 + weight4);
        } else {
            return weight3;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex3(int64_t i, int64_t width, T1 weight4)
    {
        if (width == MIN_SIZE || i <= 0) {
            return 0.0;
        } else if (out_of_range(i, width) || on_board(i, width)) {
            return 0.0;
        } else {
            return weight4;
        }
    }

    template <typename T1>
    __aicore__ inline T1 calWeights1(T1 x)
    {
        float res = ((T1)1.25 * x - (T1)2.25) * x * x + (T1)1.0;
        return res;
    }

    template <typename T1>
    __aicore__ inline T1 calWeights2(T1 x)
    {
        float res = (((T1)-0.75 * x + (T1)3.75) * x - (T1)6.0) * x + (T1)3.0;
        return res;
    }

    __aicore__ inline bool out_of_range(int64_t x, int64_t width) { return x >= (width - MIN_SIZE); };

    __aicore__ inline bool on_board(int64_t x, int64_t width)
    {
        if (x >= (width - TWO_SIZE) && x < (width - MIN_SIZE)) {
            return true;
        } else {
            return false;
        }
    };

    template <typename T1>
    __aicore__ inline T1 Min(T1 m, T1 n)
    {
        return m < n ? m : n;
    };

    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };

    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };

    __aicore__ inline void ParseTilingData(const UpsampleBicubic2dTilingData* tilingData);

    __aicore__ inline void CalculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void CalculateRatioTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void CalculateConvolution(int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);
    __aicore__ inline void ClearGM();
    __aicore__ inline void BicubicComputeBatch();
    __aicore__ inline void BicubicComputeTail();
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CubicInterp2d(int64_t indexW, int64_t indexH, int64_t offsetW, int64_t offsetH);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xIntQueueW;
    TBuf<QuePosition::VECCALC> xMinQueueW;
    TBuf<QuePosition::VECCALC> xVQueueW;
    TBuf<QuePosition::VECCALC> ratioQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xIntQueueH;
    TBuf<QuePosition::VECCALC> xMinQueueH;
    TBuf<QuePosition::VECCALC> xVQueueH;
    TBuf<QuePosition::VECCALC> ratioQueueH;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    TBuf<TPosition::VECCALC> cacheTensorBuff;
    TBuf<TPosition::VECCALC> castInputBuff;
    TBuf<TPosition::VECCALC> castOutputBuff;
    TBuf<TPosition::VECCALC> clearTensorBuff;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    LocalTensor<float> xMinTensorW;
    LocalTensor<float> xMinTensorH;
    LocalTensor<float> ratioTensorW;
    LocalTensor<float> ratioTensorH;
    LocalTensor<float> cacheTensor;
    LocalTensor<float> castInputTensor;
    LocalTensor<float> castOutputTensor;
    LocalTensor<T> clearTensor;

    int64_t blockIdx = 0;
    int64_t slideSize = 0;
    float scaleW;
    float scaleH;
    bool alignCorners;
    int64_t dataType;

    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int32_t blockSize = 8;
    int64_t startIdxW;
    int64_t startIdxH;
    int64_t batchLength;
    int64_t needCoreNum;

    uint32_t maxDataCount = DEFAULT_UB_MAX_DATA_COUNT;
};

} // namespace UpsampleBicubic2d

#endif
