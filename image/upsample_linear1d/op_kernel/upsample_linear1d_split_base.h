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
 * \file upsample_linear1d_split_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_LINEAR1D_SPLIT_BASE_H_
#define _ASCENDC_UPSAMPLE_LINEAR1D_SPLIT_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_linear_common.h"

namespace UpsampleLinear1d {
using namespace AscendC;
constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleLinear1dND {
public:
    TPipe* pipe = nullptr;
    matmul::MatmulImpl<matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                       matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                       matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                       matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, MDL_CFG>
        matmulW;

    __aicore__ inline UpsampleLinear1dND(TPipe* pipeIn) { pipe = pipeIn; };
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleLinear1dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingCommon(const UpsampleLinear1dTilingData* tilingData);
    __aicore__ inline void ParseTilingAIV();
    __aicore__ inline void ParseTilingAIC(const UpsampleLinear1dTilingData* tilingData);
    __aicore__ inline void WDirectionExpansion(int64_t startNum, int64_t endNum, bool isRemainder);
    __aicore__ inline void RowLoopFunc(int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmBlockTail,
                                       int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void RowLoopFuncRemainder(int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmBlockTail,
                                                int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void calculateWidthExtension(int64_t rowNum);
    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline void getSlideRange(const UpsampleLinear1dTilingData* tilingData);
    __aicore__ inline void PreLoad(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void AfterMatMul(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void part1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t loopT,
                                 int64_t tailNum);
    __aicore__ inline void part3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t index,
                                 int16_t length, int64_t loopT, int64_t tailNum);
    __aicore__ inline void doAicMM();
    __aicore__ inline void calculateRadio(int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK,
                                          float scale_w, bool align_corners, int64_t wIn, int64_t slide_size_w);
    __aicore__ inline void aicLoop(int64_t start, int64_t end, int64_t loopNum, int64_t tailNum);

private:
    // 系数矩阵下标队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;
    TBuf<TPosition::VECCALC> inputBuf;
    TBuf<TPosition::VECCALC> outputBuf;

    TCubeTiling matmulTiling_w;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<float> intermediateTensorGm;

    bool align_corners = false;
    bool isAicAvilable = false;
    int64_t subIdx = 0;
    int64_t blockIdx = 0;
    int64_t aicIdx = 0;
    int64_t slide_size_w = 0;
    float scale_w;

    int64_t need_core_num_w;
    int64_t need_core_num_aic;

    uint32_t radio_matrix_size_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;

    int64_t inputW = 0;
    int64_t outputW = 0;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;

    int64_t mPerTime;
    int64_t loopTimes;
    int64_t loopTail;
    int64_t inputUbSize;
    int64_t outputUbSize;
    int64_t numPerBlock;
    int64_t matmulLoopTimes;
    int64_t matmulBlockTail;
    int64_t matmulBlockPerTime;

    int64_t loopTailTimes;
    int64_t loopTailTail;
    int64_t remainderMatmulLoopTimes;
    int64_t remainderMatmulBlockTail;
    int64_t remainderLoopTailTimes;
    int64_t remainderLoopTailTail;

    int64_t inputWorkStartOffset_0 = 0;
    int64_t outputWorkStartOffset_0 = 0;
    int64_t inputH = 0;
    int64_t totalPerCore = 0;
    int64_t singleCoreN = 0;
    int64_t inputWorkStartOffsetAic = 0;
    int64_t outputWorkStartOffsetAic = 0;
    int64_t mmInputNum = 0;
    int64_t remainder_aiv_1_calc_num = 0;
    int64_t remainder_mm = 0;
    int64_t matmul_block_0_num = 0;
    int64_t matmul_block_1_num = 0;
    int64_t remainder_matmul_block_0_num = 0;
    int64_t remainderLoopTailTail_0 = 0;
    int64_t mm_tail_0 = 0;
    int64_t remainder_matmul_tail_0 = 0;
};

} // namespace UpsampleLinear1d

#endif
