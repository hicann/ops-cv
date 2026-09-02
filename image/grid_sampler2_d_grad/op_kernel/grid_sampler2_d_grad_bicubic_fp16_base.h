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
 * \file grid_sampler2_d_grad_bicubic_fp16_base.h
 * \brief
 */

#ifndef _ASCENDC_GRID_SAMPLER2_D_GRAD_BICUBIC_FP16_BASE_H_
#define _ASCENDC_GRID_SAMPLER2_D_GRAD_BICUBIC_FP16_BASE_H_

#include "kernel_operator.h"

using namespace AscendC;

constexpr static int32_t INT_MAX_FP16 = 2147483647;
constexpr static int32_t INT_MIN_FP16 = -2147483648;

template <typename T, typename Dtype, typename GridSamplerGradTilingData>
class GridSampler2DGradBicubicFP16 {
public:
    __aicore__ inline GridSampler2DGradBicubicFP16(){};
    __aicore__ inline void Init(const GridSamplerGradTilingData& __restrict tilingData,
                                GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1]);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void InitBicubicLocalTensor();
    __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount, const int32_t inputIndex);
    __aicore__ inline void CopyOut(const int32_t offset, const int32_t calCount);
    __aicore__ inline void Compute(const int32_t computeCount, const int64_t curGridPointIndex);
    __aicore__ inline void Process();

    // cubic convolution functions
    __aicore__ inline void CubicConvolution1Grad(LocalTensor<T> coeff, LocalTensor<T> x, const int32_t calCount);
    __aicore__ inline void CubicConvolution2Grad(LocalTensor<T> coeff, LocalTensor<T> x, const int32_t calCount);
    __aicore__ inline void CubicConvolution1(LocalTensor<T> coeff, LocalTensor<T> x, const int32_t calCount);
    __aicore__ inline void CubicConvolution2(LocalTensor<T> coeff, LocalTensor<T> x, const int32_t calCount);
    __aicore__ inline void GetCubicUpsampleCoefficients(LocalTensor<T> coeffTx0, LocalTensor<T> coeffTx1,
                                                        LocalTensor<T> coeffTx2, LocalTensor<T> coeffTx3,
                                                        LocalTensor<T> coeffTy0, LocalTensor<T> coeffTy1,
                                                        LocalTensor<T> coeffTy2, LocalTensor<T> coeffTy3,
                                                        LocalTensor<T> cubicTx, LocalTensor<T> cubicTy,
                                                        const int32_t calCount);

    // coordinate functions (same as bilinear)
    __aicore__ inline void ComputeSourceIndexSetGrad(LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size,
                                                     const int32_t calCount);
    __aicore__ inline void DupValue();
    __aicore__ inline T ReflectCoordinatesCommonFp16(T coord, int32_t size_val, bool align_corners_flag);
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> dataInQueue[INPUT_NUM];
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> dataOutQueue[OUTPUT_NUM];

    TBuf<TPosition::VECCALC> inputCoordinateBuf;
    TBuf<TPosition::VECCALC> xCoordinateBuf;
    TBuf<TPosition::VECCALC> yCoordinateBuf;
    TBuf<TPosition::VECCALC> xGradInBuf;
    TBuf<TPosition::VECCALC> yGradInBuf;

    // 4x4 neighborhood float coordinates
    TBuf<TPosition::VECCALC> ixNeBuf;
    TBuf<TPosition::VECCALC> iyNeBuf;
    TBuf<TPosition::VECCALC> ixNwBuf;
    TBuf<TPosition::VECCALC> iyNwBuf;
    TBuf<TPosition::VECCALC> ixSeBuf;
    TBuf<TPosition::VECCALC> iySeBuf;
    TBuf<TPosition::VECCALC> ixSwBuf;
    TBuf<TPosition::VECCALC> iySwBuf;

    // integer coordinates
    TBuf<TPosition::VECCALC> ixNeIntBuf;
    TBuf<TPosition::VECCALC> iyNeIntBuf;
    TBuf<TPosition::VECCALC> ixNwIntBuf;
    TBuf<TPosition::VECCALC> iyNwIntBuf;
    TBuf<TPosition::VECCALC> ixSeIntBuf;
    TBuf<TPosition::VECCALC> iySeIntBuf;
    TBuf<TPosition::VECCALC> ixSwIntBuf;
    TBuf<TPosition::VECCALC> iySwIntBuf;

    // cubic coefficients
    TBuf<TPosition::VECCALC> coeffTy0Buf;
    TBuf<TPosition::VECCALC> coeffTy1Buf;
    TBuf<TPosition::VECCALC> coeffTy2Buf;
    TBuf<TPosition::VECCALC> coeffTy3Buf;
    TBuf<TPosition::VECCALC> coeffTx0Buf;
    TBuf<TPosition::VECCALC> coeffTx1Buf;
    TBuf<TPosition::VECCALC> coeffTx2Buf;
    TBuf<TPosition::VECCALC> coeffTx3Buf;

    TBuf<TPosition::VECCALC> weightBuf;

    // temporary buffers
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> tmp3Buf;
    TBuf<TPosition::VECCALC> tmp4Buf;
    TBuf<TPosition::VECCALC> tmp7Buf;
    TBuf<TPosition::VECCALC> tmp8Buf;
    TBuf<TPosition::VECCALC> tmp9Buf;

    TBuf<TPosition::VECCALC> mask1Buf;
    TBuf<TPosition::VECCALC> mask2Buf;
    TBuf<TPosition::VECCALC> mask3Buf;

    TBuf<TPosition::VECCALC> selBuf1;
    TBuf<TPosition::VECCALC> selBuf2;
    TBuf<TPosition::VECCALC> selBuf3;
    TBuf<TPosition::VECCALC> selBuf4;
    TBuf<TPosition::VECCALC> dupOneBuf;

    TBuf<TPosition::VECCALC> computeIndexBuf1;
    TBuf<TPosition::VECCALC> computeIndexBuf2;
    TBuf<TPosition::VECCALC> computeIndexBuf3;
    TBuf<TPosition::VECCALC> computeIndexBuf4;
    TBuf<TPosition::VECCALC> computeIndexBuf5;
    TBuf<TPosition::VECCALC> computeIndexBuf6;
    TBuf<TPosition::VECCALC> computeIndexBuf7;
    TBuf<TPosition::VECCALC> computeIndexBuf8;
    TBuf<TPosition::VECCALC> computeIndexBuf9;

    TBuf<TPosition::VECCALC> clipLimitBuf;
    TBuf<TPosition::VECCALC> gixBuf;
    TBuf<TPosition::VECCALC> giyBuf;
    TBuf<TPosition::VECCALC> sumXBuf;
    TBuf<TPosition::VECCALC> sumYBuf;

    // fp16-specific buffers
    TBuf<TPosition::VECCALC> dstLocalBuf;
    TBuf<TPosition::VECCALC> inputXLocalTensorBuf;
    TBuf<TPosition::VECCALC> gOutLocalTensorBuf;

    GlobalTensor<Dtype> inputGm[INPUT_NUM + OUTPUT_NUM];
    GlobalTensor<T> inputGmT;

    T fheight = 0;
    T fwidth = 0;
    uint32_t batch = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    int32_t channel = 0;
    int32_t alignChannel = 0;
    int32_t height = 0;
    int32_t width = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t alignCorners = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t perBlockCount = 0;
    uint32_t blockIdx = 0;
    uint32_t dataCount = 0;
    uint32_t baseOffset = 0;
    uint32_t batchOffset = 0;
    uint32_t alignBufferNum = 0;
    uint32_t gradStrideC = 0;
    uint32_t gradStrideH = 0;
    uint32_t gradStrideW = 0;
    uint32_t xStrideC = 0;
    uint32_t dxStrideN = 0;
    uint32_t dxStrideC = 0;
    int32_t dxStrideH = 0;
    uint32_t dxStrideW = 0;
    uint32_t maskSize = 0;
    uint32_t maskNum = 0;
    uint32_t inputStrideN = 0;
    int32_t inputStrideH = 0;
    uint32_t inputStrideW = 0;
    int64_t pointIndex = 0;
    int64_t baseGradGmOffset = 0;
    int64_t baseGmOffset = 0;
    int64_t gradGmOffset = 0;
    int32_t pointOffset = 0;
    int32_t ncOffset = 0;
    int64_t xGmOffset = 0;
    int32_t group = 0;
    uint32_t tailBNum = 0;
    uint32_t isDeterministic = 0;
    T giy = static_cast<T>(0);
    T gix = static_cast<T>(0);
    RoundMode mode;

    LocalTensor<uint16_t> int8ToInt16Mask1;
    LocalTensor<uint16_t> int8ToInt16Mask2;
    LocalTensor<uint8_t> mask1Tensor;
    LocalTensor<uint8_t> mask2Tensor;
    LocalTensor<uint8_t> mask3Tensor;
    LocalTensor<T> selTensor1;
    LocalTensor<T> selTensor2;
    LocalTensor<T> selTensor3;
    LocalTensor<T> selTensor4;
    LocalTensor<T> dupOneTensor;
    LocalTensor<T> tmp1Tensor;
    LocalTensor<T> tmp2Tensor;
    LocalTensor<int32_t> tmpIndex;
    LocalTensor<T> clipLimit;
    LocalTensor<T> gixLocalTensor;
    LocalTensor<T> giyLocalTensor;
    LocalTensor<T> sumX;
    LocalTensor<T> sumY;
};

#endif // GRID_SAMPLER_2D_GRAD_BICUBIC_FP16_H_
