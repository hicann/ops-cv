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
 * \file upsample_bicubic2d_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BICUBIC2D_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BICUBIC2D_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG_BICUBIC = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;
constexpr uint32_t ADDR_ALIGN_SIZE = 128;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr int8_t MIN_SIZE = 1;
constexpr int8_t TWO_SIZE = 2;

template <typename T>
class UpsampleBicubic2dND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_BICUBIC>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_BICUBIC>
        matmulH;

    __aicore__ inline UpsampleBicubic2dND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleBicubic2dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 x, int64_t m, int64_t n, int64_t width)
    {
        float weight1 = 0;
        float weight2 = 0;
        float weight3 = 0;
        float weight4 = 0;
        float t = (float)1.0 - x;
        switch (n) {
            case 0:
                weight1 = calWeights2(1 + x);
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                return getWeightIndex0(m, width, weight1, weight2, weight3);
            case 1:
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex1(m, width, weight2, weight3, weight4);
            case 2:
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex2(m, width, weight3, weight4);
            case 3:
                weight4 = calWeights2(1 + t);
                return getWeightIndex3(m, width, weight4);
            default:
                return 0.0;
        }
    };

    template <typename T1>
    __aicore__ inline T1 getWeightIndex0(int64_t x, int64_t width, T1 weight1, T1 weight2, T1 weight3)
    {
        if (width == MIN_SIZE) {
            return 1.0;
        } else if (x < 0) {
            return (weight1 + weight2 + weight3);
        } else if (x == 0) {
            return (weight1 + weight2);
        } else if (out_of_range(x, width)) {
            return weight1;
        } else if (on_board(x, width)) {
            return weight1;
        } else {
            return weight1;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex1(int64_t x, int64_t width, T1 weight2, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE) {
            return 0.0;
        } else if (x < 0) {
            return weight4;
        } else if (x == 0) {
            return (width == TWO_SIZE) ? (weight3 + weight4) : weight3;
        } else if (out_of_range(x, width)) {
            return (weight2 + weight3 + weight4);
        } else if (on_board(x, width)) {
            return weight2;
        } else {
            return weight2;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex2(int64_t x, int64_t width, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE || x < 0) {
            return 0.0;
        } else if (x == 0) {
            return (width == TWO_SIZE) ? static_cast<float>(0.0) : weight4;
        } else if (out_of_range(x, width)) {
            return 0.0;
        } else if (on_board(x, width)) {
            return (weight3 + weight4);
        } else {
            return weight3;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex3(int64_t x, int64_t width, T1 weight4)
    {
        if (width == MIN_SIZE || x <= 0) {
            return 0.0;
        } else if (out_of_range(x, width) || on_board(x, width)) {
            return 0.0;
        } else {
            return weight4;
        }
    }

    template <typename T1>
    __aicore__ inline T1 calWeights1(T1 m)
    {
        float res = ((T1)1.25 * m - (T1)2.25) * m * m + (T1)1.0;
        return res;
    }

    template <typename T1>
    __aicore__ inline T1 calWeights2(T1 m)
    {
        float res = (((T1)-0.75 * m + (T1)3.75) * m - (T1)6.0) * m + (T1)3.0;
        return res;
    }

    __aicore__ inline bool out_of_range(int64_t m, int64_t width) { return m >= (width - MIN_SIZE); };

    __aicore__ inline bool on_board(int64_t m, int64_t width)
    {
        if (m >= (width - TWO_SIZE) && m < (width - MIN_SIZE)) {
            return true;
        } else {
            return false;
        }
    };

    template <typename T1>
    __aicore__ inline T1 Min(T1 x, T1 y)
    {
        return x < y ? x : y;
    };

    template <typename T1>
    __aicore__ inline T1 Max(T1 x, T1 y)
    {
        return x > y ? x : y;
    };

    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = static_cast<float>(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };

    __aicore__ inline void ParseTilingData(const UpsampleBicubic2dTilingData* tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();

    __aicore__ inline void calculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void calculateRatioTensorW(int64_t index, int64_t length);
    __aicore__ inline void calculateRatioTensorH(int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRatioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRatioTensor(int8_t direction);
    __aicore__ inline void releaseRatioTensor(int8_t direction, LocalTensor<T> ratioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    // 系数矩阵下标队列
    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xIntQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TBuf<QuePosition::VECCALC> xVQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ratioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xIntQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TBuf<QuePosition::VECCALC> xVQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ratioQueue_h;

    const TCubeTiling* __restrict matmulTiling_w;
    const TCubeTiling* __restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xIntTensor;
    LocalTensor<float> xVTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;
    float scale_w;
    float scale_h;
    bool align_corners;
    int64_t max_interp_size_w = 16;
    int64_t max_interp_size_h;
    int64_t need_core_num_w;
    int64_t need_core_num_h;
    int64_t dataType;

    bool floatEqual_h;
    bool floatEqual_w;

    uint64_t intermediate_matrix_size;
    uint32_t ratio_matrix_size_w;
    uint32_t ratio_matrix_size_h;

    int64_t slideStart_w;
    int64_t slideEnd_w;
    int64_t tailSlideStart_w;
    int64_t tailSlideEnd_w;
    int64_t tailRowStart_w;
    int64_t tailRowEnd_w;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRatioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;
};

} // namespace UpsampleBicubic2d

#endif
