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
 * \file upsample_bilinear2d_aa_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BILINEAR2D_AA_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BILINEAR2D_AA_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBilinear2dAA {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG_AA = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr uint32_t ADDR_ALIGN_SIZE = 128;

template <typename T>
class UpsampleBilinearAAND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_AA>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG_AA>
        matmulH;

    __aicore__ inline UpsampleBilinearAAND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleBilinearAATilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 x, T2 y)
    {
        if (y == 0) {
            return x;
        }
        return (x + y - 1) / y;
    };
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 m)
    {
        if (m < 0) {
            m = -1 * m;
        }
        if (m < (float)1.0) {
            return (float)1.0 - m;
        }
        return 0.0;
    };
    __aicore__ inline bool FloatEqual(float m, float n)
    {
        float closeTo0 = static_cast<float>(1e-6);
        if (m > n) {
            return m - n < closeTo0;
        } else {
            return n - m < closeTo0;
        }
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };
    __aicore__ inline void ParseTilingData(const UpsampleBilinearAATilingData* tilingData);
    __aicore__ inline void WExpansion();
    __aicore__ inline void HExpansion();
    __aicore__ inline void calculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void calculateRadioTensorW(int64_t index, int64_t length, float invscale);
    __aicore__ inline void calculateRadioTensorH(int64_t index, int64_t length, float invscale);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);
    __aicore__ inline void getSlideRange();

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    // 系数矩阵下标队列

    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TBuf<QuePosition::VECCALC> xSizeQueue_w;
    TBuf<QuePosition::VECCALC> weightQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TBuf<QuePosition::VECCALC> xSizeQueue_h;
    TBuf<QuePosition::VECCALC> weightQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_h;

    TBuf<QuePosition::VECCALC> floorQueue_w;
    TBuf<QuePosition::VECCALC> floorQueue_h;

    const TCubeTiling* __restrict matmulTiling_w;
    const TCubeTiling* __restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xSizeTensor;
    LocalTensor<float> weightTensor;
    LocalTensor<float> floorTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t slide_size = 0;
    int64_t blockIdx = 0;
    float scale_w;
    float scale_h;
    float invscale_w;
    float invscale_h;
    float support_w;
    float support_h;
    int64_t max_interp_size_w = 16;
    int64_t max_interp_size_h;
    int64_t need_core_num_h;
    int64_t need_core_num_w;
    int64_t dataType;

    uint64_t intermediate_matrix_size;
    uint32_t radio_matrix_size_h;
    uint32_t radio_matrix_size_w;

    int64_t slideNumW;
    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t slideNumH;
    int64_t eachCoreSlideNumH;
    int64_t tailStartSlideNumH;
    int64_t groupCoreNumH;
    int64_t tailAvergingRowsH;
    int64_t remainderH;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;

    int64_t slideStart_h = 0;
    int64_t slideEnd_h = 0;
    int64_t tailRowStart_h = 0;
    int64_t tailRowEnd_h = 0;
    int64_t tailSlideStart_h = 0;
    int64_t tailSlideEnd_h = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;
};

} // namespace UpsampleBilinear2dAA

#endif
