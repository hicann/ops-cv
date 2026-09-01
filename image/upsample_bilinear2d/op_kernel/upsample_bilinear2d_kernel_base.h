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
 * \file upsample_bilinear2d_kernel_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_BILINEAR2D_KERNEL_BASE_H_
#define _ASCENDC_UPSAMPLE_BILINEAR2D_KERNEL_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_bilinear2d_common.h"

namespace UpsampleBilinear2d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr uint32_t ADDR_ALIGN_SIZE = 128;

template <typename T>
class UpsampleBilinear2dND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulH;

    __aicore__ inline UpsampleBilinear2dND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                UpsampleBilinear2dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(UpsampleBilinear2dTilingData* tilingData);
    __aicore__ inline void ExpansionW();
    __aicore__ inline void ExpansionH();
    __aicore__ inline void calculateRadioTensorW(int64_t loopIndex, int64_t length);
    __aicore__ inline void calculateRadioTensorH(int64_t loopIndex, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);
    __aicore__ inline void getSlideRange();

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    TBuf<TPosition::VECCALC> UbBuf;

    // 系数矩阵下标队列
    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_h;

    const TCubeTiling* __restrict matmulTiling_w;
    const TCubeTiling* __restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    bool align_corners = false;
    int64_t blockIdx = 0;
    int64_t slide_size_w = 0;
    int64_t slide_size_h = 0;
    float scale_w;
    float scale_h;

    float support = 1.0;
    int64_t max_interp_size = 2;

    int64_t need_core_num_w;
    int64_t need_core_num_h;
    int64_t dataType;

    uint64_t intermediate_matrix_size;
    uint32_t radio_matrix_size_h;
    uint32_t radio_matrix_size_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t eachCoreSlideNumH;
    int64_t tailStartSlideNumH;
    int64_t slideNumH;
    int64_t groupCoreNumH;
    int64_t tailAvergingRowsH;
    int64_t remainderH;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;

    int64_t slideStart_h = 0;
    int64_t slideEnd_h = 0;
    int64_t tailSlideStart_h = 0;
    int64_t tailSlideEnd_h = 0;
    int64_t tailRowStart_h = 0;
    int64_t tailRowEnd_h = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;

    int64_t wInMaxIdx = 0;
    int64_t hInMaxIdx = 0;
};

} // namespace UpsampleBilinear2d

#endif
