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
 * \file upsample_nearest3d_310p_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_NEAREST3D_310P_BASE_H_
#define _ASCENDC_UPSAMPLE_NEAREST3D_310P_BASE_H_

#include <type_traits>
#include "upsample_nearest3d_struct.h"
#include "kernel_operator.h"

namespace UpsampleNearest3d {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int8_t D_INDEX = 0;
constexpr int8_t H_INDEX = 1;
constexpr int8_t W_INDEX = 2;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr float BEST_PERFORMANCE_SCALE = 100.0f;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

const int64_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;
const int64_t DEFAULT_SYNC_UB_SIZE = 1 * 1024;

template <typename T>
class UpsampleNearest3dND310p {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearest3dND310p(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, bool isNearestExact, GM_ADDR workspace,
                                const UpsampleNearest3dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 x, T2 y)
    {
        if (y == 0) {
            return x;
        }
        return (x + y - 1) / y;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 x, T1 y)
    {
        return x > y ? x : y;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    __aicore__ inline void ClearGM();
    __aicore__ inline void ParseTilingData(const UpsampleNearest3dTilingData* tilingData);
    __aicore__ inline void GatherData(int64_t slideIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyParams repeatParams);
    __aicore__ inline void ComputeAndCopyOut(uint32_t dataCount, uint32_t srcDataLength, uint32_t blockCount,
                                             int64_t outputOffset);
    __aicore__ inline void CopyOutProcess(int64_t offsetTemp, LocalTensor<T> dstLocal);
    __aicore__ inline void CopyOut(int64_t offsetTemp, LocalTensor<T> dstLocal, int64_t copyOutCnt);
    __aicore__ inline void GetRangeW(int64_t slideIndex);
    __aicore__ inline void GetRangeH(int64_t slideIndex);
    __aicore__ inline void GetRangeD(int64_t slideIndex);
    __aicore__ inline void CalculateSrcIndexTensor(int64_t index, int64_t length, int8_t direction,
                                                   LocalTensor<float> srcIndexTensor);
    __aicore__ inline void CalculateGatherOffsetW();

private:
    TBuf<QuePosition::VECCALC> srcIndexQueueW;
    TBuf<QuePosition::VECCALC> srcIndexQueueH;
    TBuf<QuePosition::VECCALC> srcIndexQueueD;
    TBuf<QuePosition::VECCALC> srcOffsetQueue;
    TBuf<TPosition::VECCALC> clearTensorBuff;
    TBuf<TPosition::VECCALC> syncTensorBuff;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> workQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<int32_t> syncGM;

    LocalTensor<float> srcIndexTensorW;
    LocalTensor<float> srcIndexTensorH;
    LocalTensor<float> srcIndexTensorD;
    LocalTensor<int32_t> srcOffsetTensor;
    LocalTensor<uint32_t> gatherOffsetTensor;
    LocalTensor<float> cacheTensor;

    int64_t blockIdx = 0;
    bool isExact = false;
    int64_t batches = 0;
    int64_t inputShapes[3] = {0};
    int64_t outputShapes[3] = {0};
    float scales[3] = {ZERO_FLOAT};

    int64_t slideSizeW = 0;
    int64_t tensorSizeW = 0;
    int64_t tensorSizeH = 0;
    int64_t tensorSizeD = 0;

    int64_t slideNumH = 0;
    int64_t slideNumD = 0;
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    int64_t tailStartSlideNum = 0;
    int64_t groupCoreNum = 0;
    int64_t inputRow = 0;
    int64_t tailAvergingRow = 0;
    int64_t needCoreNum = 0;

    int64_t lastStartW = -1;
    int64_t startW = 0;
    int64_t endW = 0;
    int64_t dataCount = 0;
    int64_t srcStartW = 0;
    int64_t srcEndW = 0;
    int64_t srcDataCount = 0;
    int64_t srcDataLength = 0;
    int64_t batchNum = 0;
    uint16_t srcBlockLen = 0;
    uint16_t srcStride = 0;

    int64_t indexH = 0;
    int64_t srcIndexH = 0;
    int64_t heightCount = 0;

    int64_t indexD = 0;
    int64_t srcIndexD = 0;
    int64_t depthCount = 0;

    int64_t blockSize = 8;
    int64_t totalNum = 0;
};

} // namespace UpsampleNearest3d

#endif
