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
 * \file upsample_bicubic2d_grad_dc_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BICUBIC2D_GRAD_DC_BASE_H_
#define _ASCENDC_UPSAMPLE_BICUBIC2D_GRAD_DC_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2dGrad {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t NUMBER_TWO = 2;
constexpr int32_t NUMBER_THREE = 3;
constexpr int32_t NUMBER_FOUR = 4;
constexpr int32_t NUMBER_SIX = 6;

constexpr int32_t DATA_BLOCK_BYTES = 32;
constexpr int32_t ONE_K_BYTES = 1024;
constexpr MatmulConfig MDL_CFG_BICUBIC_GRAD = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleBicubic2dGradDCND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_BICUBIC_GRAD>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_BICUBIC_GRAD>
        matmulH;
    __aicore__ inline UpsampleBicubic2dGradDCND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                UpsampleBicubic2dGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcWeights(float (&weights)[4], float tValue)
    {
        float x1 = tValue; // tValue 为当前中心点偏移值，x1为左侧点偏移值
        weights[0] = CalcWeight1(x1 + 1);
        weights[1] = CalcWeight2(x1);
        float x2 = 1 - tValue; // tValue 为当前中心点偏移值，x2为右侧点偏移值
        weights[NUMBER_TWO] = CalcWeight2(x2);
        weights[NUMBER_THREE] = CalcWeight1(x2 + 1); // x2为右侧点偏移值，计算第二个点偏移值
    };
    // 计算weight,可将a替换为固定值
    __aicore__ inline float CalcWeight1(float x)
    {
        constexpr float COEFFICIENT_1 = -0.75f;
        constexpr float COEFFICIENT_2 = 3.75f;
        return ((x * COEFFICIENT_1 + COEFFICIENT_2) * x - static_cast<float>(NUMBER_SIX)) * x +
               static_cast<float>(NUMBER_THREE);
    };
    __aicore__ inline float CalcWeight2(float x)
    {
        constexpr float COEFFICIENT_1 = 1.25f;
        constexpr float COEFFICIENT_2 = 2.25f;
        return (x * COEFFICIENT_1 - COEFFICIENT_2) * x * x + 1.0f;
    };
    template <typename T1, typename T2>
    __aicore__ inline auto AlignUp(T1 a, T2 b) -> decltype(a + b)
    {
        if (b <= 0) {
            return a;
        }
        auto ca = static_cast<decltype(a + b)>(a);
        auto cb = static_cast<decltype(a + b)>(b);
        if (ca % cb == 0) {
            return ca;
        }
        return (ca + cb - 1) / cb * cb;
    }
    template <typename T1, typename T2>
    __aicore__ inline auto AlignDown(T1 a, T2 b) -> decltype(a + b)
    {
        if (b <= 0) {
            return a;
        }
        auto ca = static_cast<decltype(a + b)>(a);
        auto cb = static_cast<decltype(a + b)>(b);
        if (ca % cb == 0) {
            return ca;
        }
        return ca / cb * cb;
    }
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return b == 0 ? a : ((a + b - 1) / b);
    };
    template <typename T1>
    __aicore__ inline int64_t Ceil(T1 x)
    {
        if (x < 0) {
            x = x - 1;
        }
        int64_t floor_v = int64_t(x);
        return x == floor_v ? floor_v : (floor_v + 1);
    };
    template <typename T1, typename T2>
    __aicore__ inline auto GetMin(T1 a, T2 b)
    {
        return a < b ? a : b;
    };
    template <typename T1, typename T2>
    __aicore__ inline auto GetMax(T1 a, T2 b)
    {
        return a >= b ? a : b;
    };
    __aicore__ inline void InitGlobalTensors(GM_ADDR input, GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void InitScalars();
    __aicore__ inline void InitLocalTensors();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void CalculateIntermediateTensor(int64_t xMinStart, int64_t maxIdx, float scale, int64_t length);
    __aicore__ inline int64_t CalculateInstartIdx(int64_t startIdx, float scale);
    __aicore__ inline void ParseTilingData(UpsampleBicubic2dGradTilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index, GM_ADDR tensorPtr);
    __aicore__ inline void CalculateRadioTensor(int64_t index, int64_t length, int64_t direction, int64_t slideKNum);
    __aicore__ inline void calculateWidthExtension(int64_t xMin, int64_t tensorCIndex, int64_t rowStart,
                                                   int64_t rowEnd);
    __aicore__ inline void CopyRadioTensorToGm(int64_t length, int64_t kStartIdx, int64_t slideKNum);
    __aicore__ inline void CopyRadioTensorToGmY(int64_t length, int64_t singleCoreK, int64_t kStartIdx,
                                                int64_t slideKNum);
    __aicore__ inline void calculateHeightExtension(int64_t xMin, int64_t tensorCIndex, int64_t rowStart,
                                                    int64_t rowEnd);
    __aicore__ inline void InitEventId();

private:
    TBuf<QuePosition::VECCALC> ubBuf;

    const TCubeTiling* __restrict matmulTilingW;
    const TCubeTiling* __restrict matmulTilingH;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xTensor;
    LocalTensor<float> tTensor;

    LocalTensor<float> radioTensor;
    LocalTensor<T> radioCopyOutTensor;

    event_t eventIDVToS;
    event_t eventIDSToV;
    event_t eventIdMTE3ToMTE2;
    event_t eventIdSToMTE3;
    event_t eventIdVToMTE3;
    event_t eventIdMTE3ToV;
    event_t eventIdMTE3ToS;

    int64_t ubMaxBytes = 0;

    int64_t xMin = 0;
    int64_t aiCoreIdx = 0;
    int64_t blockIdx = 0;
    int64_t slideSize = 0;

    int64_t alignCorners = 0;
    float scaleW;
    float scaleH;

    uint64_t intermediateMatrixSize = 16;
    int64_t radioMatrixSize;
    // 切分块在原系数矩阵中的位置
    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    // 系数矩阵切块的宽度
    int64_t slidelenW;
    int64_t slidelenH;

    int64_t slideStartH;
    int64_t slideEndH;
    int64_t tailSlideStartH;
    int64_t tailSlideEndH;
    int64_t tailRowStartH;
    int64_t tailRowEndH;

    float realScaleW = 0;
    float realScaleH = 0;
    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};

    uint32_t needCoreNumW;
    uint32_t needCoreNumH;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreMaxKW = 0;
    int64_t singleCoreMaxKH = 0;
    int64_t singleCoreKW = 0;
    int64_t singleCoreKH = 0;

    bool needExpandW = false;
    bool needExpandH = false;

    bool isZeroVecCore = false; // 当前是否为aicore的第0个vector核

    int64_t splitSingleCoreKMax = 0; // 单次切K的最大值
    int64_t perDataBlockNum = 1;     // 每个dataBlock内的T类型数据量
    int64_t maxRadioBytes = 0;       // 系数矩阵的最大内存
};

} // namespace UpsampleBicubic2dGrad

#endif
