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
 * \file upsample_bicubic2d_aa_grad_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BICUBIC2D_AA_GRAD_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BICUBIC2D_AA_GRAD_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpSampleBicubic2dAAGrad {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class UpSampleBicubic2dAAGradND {
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
    __aicore__ inline UpSampleBicubic2dAAGradND(){};
    __aicore__ inline void calculateIntermediateTensorX(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
                                                        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor,
                                                        int64_t slideStart_w, int64_t slideEnd_w);
    __aicore__ inline void calculateIntermediateTensorY(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
                                                        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor,
                                                        int64_t slideStart_h, int64_t slideEnd_h);
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                UpsampleBicubicAAGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = static_cast<float>(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };
    template <typename T1>
    __aicore__ inline T1 getWeight(T1 x)
    {
        if (x < 0) {
            x = -1 * x;
        }
        if (x < (float)1.0) {
            return ((float)1.5 * x - (float)2.5) * x * x + (float)1.0;
        }
        return (((float)2.5 - (float)0.5 * x) * x - (float)4.0) * x + (float)2.0;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 x1, T2 x2)
    {
        if (x2 == 0) {
            return x1;
        }
        return (x1 + x2 - 1) / x2;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1>
    __aicore__ inline T1 getMax(T1 m, T1 n)
    {
        if (m >= n) {
            return m;
        } else {
            return n;
        }
    }
    template <typename T1>
    __aicore__ inline T1 getMin(T1 x, T1 y)
    {
        if (x >= y) {
            return y;
        } else {
            return x;
        }
    }
    __aicore__ inline void getQueueSize();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void computeIndexValueH(LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h,
                                              int64_t index, int64_t length);
    __aicore__ inline void computeIndexValueW(LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor,
                                              int64_t index, int64_t length);
    __aicore__ inline void ParseTilingData(UpsampleBicubicAAGradTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float>& float32Tensor);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline void ComputeAndCopyOut(int64_t index, int64_t dataCount, LocalTensor<float>& float32Tensor);
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index, GM_ADDR tensorPtr);
    __aicore__ inline void calculateRadioTensor(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
                                                LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor,
                                                int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorH(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
                                                 LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor,
                                                 int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm();
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;

    // 系数矩阵下标队列,横轴和纵轴范围
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> xMinQueue;
    TBuf<QuePosition::VECCALC> xSizeQueue;
    TBuf<QuePosition::VECCALC> floorQueue;
    TBuf<QuePosition::VECCALC> weightQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioCastQueue;

    const TCubeTiling* __restrict matmulTiling_w;
    const TCubeTiling* __restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t slide_size = 0;
    int64_t blockIdx = 0;

    float scale_h;
    float scale_w;
    float invscale_h;
    float invscale_w;
    float support_h;
    float support_w;
    int64_t max_interp_size_h;
    int64_t max_interp_size_w;

    uint64_t intermediate_matrix_size = 16;
    uint32_t radio_matrix_size;
    uint32_t radio_matrix_size_h;
    // 切分块在原系数矩阵中的位置
    int64_t slideStart_w;
    int64_t slideEnd_w;
    int64_t tailSlideStart_w;
    int64_t tailSlideEnd_w;
    int64_t tailRowStart_w;
    int64_t tailRowEnd_w;

    // 系数矩阵切块的宽度
    int64_t slidelen;
    int64_t slidelen_h;
    int64_t queueSize = 0;
    int64_t queueSizeW = 0;
    int64_t queueSizeH = 0;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;
    int64_t dataType;

    float zeroScaleW = 0;
    float zeroScaleH = 0;
    int64_t tensorRemainerStartOffset_w;
    int64_t tensorRemainerEndOffset_w;
    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    uint64_t inputsTensorUbSize = 0;
    int64_t* tensorDataCountList = nullptr;
    uint32_t tensorStart_w = {0};
    uint32_t tensorEnd_w = {0};
    int64_t tensorStartOffset_w = {0};
    int64_t tensorEndOffset_w = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t instart_w = 0;
    int64_t instart_h = 0;

    int64_t xMin = 0;
    int64_t instartIndex = 0;
    int64_t inendIndex = 0;

    int32_t singleCoreK_h = 0;

    bool needExpendX = false;
    bool needExpendY = false;
};

} // namespace UpSampleBicubic2dAAGrad

#endif
