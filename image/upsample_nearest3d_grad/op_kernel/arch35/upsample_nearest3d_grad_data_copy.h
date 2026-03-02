/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest3d_grad_data_copy.h
 * \brief upsample_nearest3d_grad_data_copy.h
 */
#ifndef UPSAMPLE_NEAREST3D_GRAD_DATA_COPY_H
#define UPSAMPLE_NEAREST3D_GRAD_DATA_COPY_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest3d_grad_tiling_data.h"

namespace UpsampleNearest3dGrad {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T1>
class Nearest3dGradDataCopy {
public:
    __aicore__ inline Nearest3dGradDataCopy(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, TPipe* pipeIn, const UpsampleNearest3dGradRegBaseTilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t xOffsetInGM, int64_t length);

    __aicore__ inline void CopyOut(int64_t yOffsetInGM, int64_t length);

private:
    const UpsampleNearest3dGradRegBaseTilingData* tilingData;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue;

    int64_t totalLength = 0;
    int64_t totalOffset = 0;
    int32_t ubFactor = 0;
    int32_t tailBlockNum = 0;
    int32_t blockIdx = 0;
    int32_t realCoreNum = 0;
    TPipe* pipe;
    GlobalTensor<uint8_t> xGM;
    GlobalTensor<uint8_t> yGM;
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyExtParams gm2ubParams{1, 1, 0, 0, 0};
};

template <typename T1>
__aicore__ inline void Nearest3dGradDataCopy<T1>::Init(
    GM_ADDR x, GM_ADDR y, TPipe* pipeIn, const UpsampleNearest3dGradRegBaseTilingData* __restrict tiling)
{
    pipe = pipeIn;
    tilingData = tiling;
    xGM.SetGlobalBuffer((__gm__ uint8_t*)x);
    yGM.SetGlobalBuffer((__gm__ uint8_t*)y);
    ubFactor = tilingData->ubFactor;
    totalLength = tilingData->blkProcessNum;
    tailBlockNum = tilingData->tailBlockNum;
    blockIdx = GetBlockIdx();
    realCoreNum = GetBlockNum();
    pipe->InitBuffer(dataQue, BUFFER_NUM, ubFactor * sizeof(T1));
}

template <typename T1>
__aicore__ inline void Nearest3dGradDataCopy<T1>::CopyIn(int64_t xOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> xTensor = dataQue.AllocTensor<uint8_t>();
    gm2ubParams.blockLen = length * sizeof(T1);
    DataCopyPad(xTensor, xGM[xOffsetInGM * sizeof(T1)], gm2ubParams, padParams);
    dataQue.EnQue(xTensor);
}

template <typename T1>
__aicore__ inline void Nearest3dGradDataCopy<T1>::CopyOut(int64_t yOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> yTensor = dataQue.DeQue<uint8_t>();
    gm2ubParams.blockLen = length * sizeof(T1);
    DataCopyPad(yGM[yOffsetInGM * sizeof(T1)], yTensor, gm2ubParams);
    dataQue.FreeTensor(yTensor);
}
__aicore__ inline int64_t Min(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

template <typename T1>
__aicore__ inline void Nearest3dGradDataCopy<T1>::Process()
{
    if (blockIdx >= realCoreNum) {
        return;
    }
    if (blockIdx < tailBlockNum) {
        totalLength += 1;
        totalOffset = blockIdx * totalLength;
    } else {
        totalOffset = blockIdx * totalLength + tailBlockNum;
    }

    for (int64_t loop = 0; loop < totalLength; loop += ubFactor) {
        int64_t length = Min(ubFactor, totalLength - loop);
        int64_t offset = totalOffset + loop;
        CopyIn(offset, length);
        CopyOut(offset, length);
    }
}
} // namespace UpsampleNearest3dGrad

#endif // UPSAMPLE_NEAREST3D_GRAD_DATA_COPY_H
