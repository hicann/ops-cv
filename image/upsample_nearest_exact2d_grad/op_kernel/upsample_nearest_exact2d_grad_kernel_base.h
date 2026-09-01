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
 * \file upsample_nearest_exact2d_grad_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_NEAREST_EXACT2D_GRAD_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_NEAREST_EXACT2D_GRAD_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpSampleNearestExact2dGrad {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class UpSampleNearestExact2dGradND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulH;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulW;
    __aicore__ inline UpSampleNearestExact2dGradND(){};
    __aicore__ inline void calculateIntermediateTensorX(LocalTensor<float> centerTensor, LocalTensor<float> downTensor,
                                                        LocalTensor<float> upTensor, int64_t slideStart_w,
                                                        int64_t slideEnd_w);
    __aicore__ inline void calculateIntermediateTensorY(LocalTensor<float> centerTensor, LocalTensor<float> downTensor,
                                                        LocalTensor<float> upTensor, int64_t slideStart_h,
                                                        int64_t slideEnd_h);
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, bool isExact, GM_ADDR workspace,
                                UpsampleNearestExact2dGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool FloatEqual(float m, float n)
    {
        float closeTo0 = static_cast<float>(1e-6);
        if (m > n) {
            return m - n < closeTo0;
        } else {
            return n - m < closeTo0;
        }
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 m, T2 n)
    {
        if (n == 0) {
            return m;
        }
        return (m + n - 1) / n;
    };

    template <typename T1>
    __aicore__ inline T1 Min(T1 m, T1 n)
    {
        return m < n ? m : n;
    };

    template <typename T1>
    __aicore__ inline T1 getMax(T1 x, T1 y)
    {
        if (x >= y) {
            return x;
        } else {
            return y;
        }
    }

    __aicore__ inline void wDirectionExpansion();
    __aicore__ inline void hDirectionExpansion();
    __aicore__ inline void ParseTilingData(UpsampleNearestExact2dGradTilingData* tilingData);
    __aicore__ inline void calculateRadioTensorW(LocalTensor<float> centerTensor, LocalTensor<float> downTensor,
                                                 LocalTensor<float> upTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorH(LocalTensor<float> centerTensor, LocalTensor<float> downTensor,
                                                 LocalTensor<float> upTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm();
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> upQueue;
    TBuf<QuePosition::VECCALC> downQueue;
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

    float scale_w;
    float scale_h;
    float invscale_h;
    float invscale_w;
    float support_w;
    float support_h;
    int64_t max_interp_size_w;
    int64_t max_interp_size_h;

    uint64_t intermediate_matrix_size = 0;
    uint32_t radio_matrix_size;
    uint32_t radio_matrix_size_h;
    uint32_t slideStart_w;
    uint32_t slideEnd_w;
    uint32_t tailRowStart_w;
    uint32_t tailRowEnd_w;
    uint32_t tailSlideStart_w;
    uint32_t tailSlideEnd_w;

    uint32_t slidelen;
    uint32_t slidelen_h;

    uint32_t slideStart_h;
    uint32_t slideEnd_h;
    uint32_t tailSlideStart_h;
    uint32_t tailSlideEnd_h;
    uint32_t tailRowStart_h;
    uint32_t tailRowEnd_h;
    int16_t dataType;

    int64_t cubeSize = 0;
    int64_t middleSize = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;

    int64_t xMin = 0;
    int64_t instartIndex = 0;
    int64_t inendIndex = 0;

    int64_t instart_w = 0;
    int64_t instart_h = 0;
    int64_t wIndex = 0;
    int64_t hIndex = 0;
    int32_t singleCoreK_h = 0;
    bool exactMode = false;
    bool isExpandH = true;
    bool isExpandW = true;
    int32_t tailBatchStart_h;
    int32_t tailBatchEnd_h;
};

} // namespace UpSampleNearestExact2dGrad

#endif
