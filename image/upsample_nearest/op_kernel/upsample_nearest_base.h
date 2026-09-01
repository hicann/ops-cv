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
 * \file upsample_nearest_base.h
 * \brief
 */

#ifndef _ASCENDC_UPSAMPLE_NEAREST_BASE_H_
#define _ASCENDC_UPSAMPLE_NEAREST_BASE_H_

#include <type_traits>
#include "kernel_operator.h"

namespace UpsampleNearest {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t NO_BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_MIN_NUM = 16;

// uint8 SmallC: row-pack when a per-column write is smaller than one 32B block.
constexpr int64_t SMALLC_PACK_C = 32;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

const int64_t DEFAULT_UB_MAX_DATA_COUNT = 2048;
const int64_t DEFAULT_UB_MAX_COPY_SIZE = 64 * 1024; // 64kb

template <typename T>
__aicore__ inline constexpr int32_t GatherElemSize()
{
    return sizeof(T) >= sizeof(half) ? sizeof(T) : sizeof(half);
}

template <typename T, int32_t MODE>
class UpsampleNearestND {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearestND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const UpsampleNearestTilingData* __restrict__ tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };

    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };

    __aicore__ inline void ParseTilingData(const UpsampleNearestTilingData* __restrict__ tilingData);

    __aicore__ inline void CalculateIdxTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void NearestComputeBase();
    __aicore__ inline void NearestComputeSmallCW();
    __aicore__ inline void NearestComputeSmallNCH();
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyInBatch(int64_t indexInput, int64_t calCount, uint16_t blockCnt);
    __aicore__ inline void CopyOutBatch(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyOutBase(LocalTensor<T> dstDataLocal, int64_t indexOutput, int64_t calCount);
    __aicore__ inline void ProcessOutput(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                         int64_t lengthH);
    __aicore__ inline void ProcessOutputBase(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                             int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallC(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                               int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallCW(int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW,
                                                int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallCWAllBatch(int64_t indexW, int64_t indexH, int64_t lengthW,
                                                        int64_t lengthH);

    __aicore__ inline void GatherWithBridge(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<uint32_t>& offset,
                                            int64_t count, int64_t srcCount = -1);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xIntQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xIntQueueH;
    TBuf<QuePosition::VECCALC> gatherQueue;
    TBuf<QuePosition::VECCALC> offsetQueue;
    TBuf<QuePosition::VECCALC> halfSrcBuf;
    TBuf<QuePosition::VECCALC> halfDstBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    int64_t blockIdx = 0;
    int64_t slideSize = 512;
    float scaleW;
    float scaleH;
    int64_t dataType;

    int64_t tailColStart;
    int64_t tailColEnd;
    int64_t tailRowStart;
    int64_t tailRowEnd;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int32_t blockSize = 8;
    int64_t inputBatchSize;
    int64_t outputBatchSize;
    bool exactMode;
    bool useU16Gather = false;

    int64_t maxCopyCount;
};

} // namespace UpsampleNearest

#endif
