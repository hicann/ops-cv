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
 * \file upsample_bilinear2d_grad_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BILINEAR2D_GRAD_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BILINEAR2D_GRAD_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpSampleBilinear2dGrad {
using namespace AscendC;
constexpr MatmulConfig MDL_CFG_GRAD = GetMDLConfig(true, false, 0, false, false, false, true);
constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t RESERVED_VALUE = 4;
constexpr float RESERVED_scale = 1.5;

template <typename T>
class UpSampleBilinear2dGradND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_GRAD>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_GRAD>
        matmulH;
    __aicore__ inline UpSampleBilinear2dGradND(){};
    __aicore__ inline void calculateIntermediateTensorX(LocalTensor<float> centerTensor,
                                                        LocalTensor<float> xIndexTensor,
                                                        LocalTensor<float> xLambdaTensor, int64_t slideStart_w,
                                                        int64_t slideEnd_w);
    __aicore__ inline void calculateIntermediateTensorY(LocalTensor<float> centerTensor,
                                                        LocalTensor<float> xIndexTensor,
                                                        LocalTensor<float> xLambdaTensor, int64_t slideStart_h,
                                                        int64_t slideEnd_h);
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                UpsampleBilinear2dGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool FloatEqual(float x, float y)
    {
        float closeTo0 = float(1e-6);
        if (x > y) {
            return x - y < closeTo0;
        } else {
            return y - x < closeTo0;
        }
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 i, T2 j)
    {
        if (j == 0) {
            return i;
        }
        return (i + j - 1) / j;
    };

    template <typename T1>
    __aicore__ inline T1 Min(T1 x, T1 y)
    {
        return x < y ? x : y;
    };

    template <typename T1>
    __aicore__ inline T1 Max(T1 x, T1 y)
    {
        if (x >= y) {
            return x;
        } else {
            return y;
        }
    }
    __aicore__ inline void setRadioValueW(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
                                          LocalTensor<float> centerTensor, int64_t index, int64_t length);

    __aicore__ inline void setRadioValueH(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
                                          LocalTensor<float> centerTensor, int64_t index, int64_t length);
    __aicore__ inline void setZeroRadioValue(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
                                             LocalTensor<float> centerTensor, int64_t index, int64_t length);
    __aicore__ inline void getQueueSize();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void ParseTilingData(UpsampleBilinear2dGradTilingData* tilingData);
    __aicore__ inline void calculateRadioTensor(LocalTensor<float> centerTensor, LocalTensor<float> xIndexTensor,
                                                LocalTensor<float> xLambdaTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorH(LocalTensor<float> centerTensor, LocalTensor<float> xIndexTensor,
                                                 LocalTensor<float> xLambdaTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm(int64_t length);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    // 系数矩阵下标队列,横轴和纵轴范围
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> xIndexQueue;
    TBuf<QuePosition::VECCALC> xLambdaQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioCastQueue;

    const TCubeTiling* __restrict matmulTiling_w;
    const TCubeTiling* __restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;
    int64_t radioSize = 0;
    float scale_w;
    float scale_h;

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
    int64_t queueSize = 0;
    int64_t cubeSize = 0;
    int64_t middleSize = 0;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;
    int64_t dataType;

    float zeroScaleW = 0;
    float zeroScaleH = 0;
    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t workSpaceRadioOffset = 0;
    int64_t workSpaceLineOffset = 0;
    int64_t singleCoreK = 0;
    int64_t instart_w = 0;
    int64_t instart_h = 0;

    int64_t xMin = 0;
    int64_t instartIndex = 0;
    int64_t inendIndex = 0;
    int64_t align_corners = 0;
    int32_t singleCoreK_h = 0;
    int32_t tailBatchStart_h = 0;
    int32_t tailBatchEnd_h = 0;
    bool needExpendW = false;
    bool needExpendH = false;
};

} // namespace UpSampleBilinear2dGrad

#endif
