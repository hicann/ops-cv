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
 * \file upsample_bilinear2d_aa_backward_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BILINEAR2D_AA_BACKWARD_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BILINEAR2D_AA_BACKWARD_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBilinear2dAABackward {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class UpsampleBilinear2dAABackwardND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulH;

    __aicore__ inline UpsampleBilinear2dAABackwardND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                UpsampleBilinear2dAABackwardTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 x)
    {
        if (x < 0) {
            x = -1 * x;
        }
        if (x < (float)1.0) {
            return (float)1.0 - x;
        }
        return 0.0;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 m, T1 n)
    {
        return m > n ? m : n;
    };
    __aicore__ inline void ParseTilingData(UpsampleBilinear2dAABackwardTilingData* tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void calculateIntermediateTensorW(int64_t index, int64_t length);
    __aicore__ inline void calculateIntermediateTensorH(int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorW(int64_t index, int64_t length, int64_t minIndex);
    __aicore__ inline void calculateRadioTensorH(int64_t index, int64_t length, int64_t minIndex);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd,
                                                   int64_t length);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd,
                                                    int64_t length);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xMinQueueW;
    TBuf<QuePosition::VECCALC> xSizeQueueW;
    TBuf<QuePosition::VECCALC> weightQueueW;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xMinQueueH;
    TBuf<QuePosition::VECCALC> xSizeQueueH;
    TBuf<QuePosition::VECCALC> weightQueueH;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueueH;

    TBuf<QuePosition::VECCALC> floorQueueW;
    TBuf<QuePosition::VECCALC> floorQueueH;

    const TCubeTiling* __restrict matmulTilingW;
    const TCubeTiling* __restrict matmulTilingH;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xSizeTensor;
    LocalTensor<float> weightTensor;
    LocalTensor<float> floorTensor;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slideSize = 0;
    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;
    float supportW;
    float supportH;
    int64_t maxInterpSizeW;
    int64_t maxInterpSizeH;
    int64_t needCoreNumW;
    int64_t needCoreNumH;
    bool needResizeW = true;
    bool needResizeH = true;

    uint8_t dataType;
    uint64_t intermediateMatrixSize;
    uint32_t radioMatrixSizeW;
    uint32_t radioMatrixSizeH;

    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    int64_t slideStartH;
    int64_t slideEndH;
    int64_t tailSlideStartH;
    int64_t tailSlideEndH;
    int64_t tailRowStartH;
    int64_t tailRowEndH;

    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};
    int64_t workSpaceRadioOffset = 0;
    int64_t xMin = 0;
    int64_t singleCoreK = 0;
};

} // namespace UpsampleBilinear2dAABackward

#endif
