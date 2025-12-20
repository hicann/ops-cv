/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bicubic2d_aa_data_copy.h
 * \brief upsample_bicubic2d_aa_data_copy.h
 */
#ifndef UPSAMPLE_BICUBIC2D_AA_DATA_COPY_H
#define UPSAMPLE_BICUBIC2D_AA_DATA_COPY_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_tiling_data.h"

namespace UpsampleBicubic2dAA {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T1>
class Bicubic2dAADataCopy {
public:
    __aicore__ inline Bicubic2dAADataCopy(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe *pipe,
        const UpsampleBicubic2dAARegBaseTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t xOffsetInGM, int64_t length);

    __aicore__ inline void CopyOut(int64_t yOffsetInGM, int64_t length);

private:
    const UpsampleBicubic2dAARegBaseTilingData *tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;

    int64_t totalLength_ = 0;
    int64_t totalOffset_ = 0;
    int32_t ubFactor_ = 0;
    int32_t tailBlockNum_ = 0;
    int32_t blockIdx_ = 0;
    int32_t realCoreNum_ = 0;
    TPipe *pipe_;
    GlobalTensor<uint8_t> xGM_;
    GlobalTensor<uint8_t> yGM_;
    DataCopyPadExtParams<uint8_t> padParams_{ false, 0, 0, 0 };
    DataCopyExtParams gm2ubParams_{ 1, 1, 0, 0, 0 };
};

template <typename T1>
__aicore__ inline void Bicubic2dAADataCopy<T1>::Init(GM_ADDR x, GM_ADDR y, TPipe *pipe,
    const UpsampleBicubic2dAARegBaseTilingData *__restrict tilingData)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    xGM_.SetGlobalBuffer((__gm__ uint8_t *)x);
    yGM_.SetGlobalBuffer((__gm__ uint8_t *)y);
    ubFactor_ = tilingData_->ubFactor;
    totalLength_ = tilingData_->blkProcessNum;
    tailBlockNum_ = tilingData_->tailBlockNum;
    blockIdx_ = GetBlockIdx();
    realCoreNum_ = GetBlockNum();
    pipe_->InitBuffer(dataQue_, BUFFER_NUM, ubFactor_ * sizeof(T1));
}

template <typename T1>
__aicore__ inline void Bicubic2dAADataCopy<T1>::CopyIn(int64_t xOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> xTensor = dataQue_.AllocTensor<uint8_t>();
    gm2ubParams_.blockLen = length * sizeof(T1);
    DataCopyPad(xTensor, xGM_[xOffsetInGM * sizeof(T1)], gm2ubParams_, padParams_);
    dataQue_.EnQue(xTensor);
}

template <typename T1>
__aicore__ inline void Bicubic2dAADataCopy<T1>::CopyOut(int64_t yOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> yTensor = dataQue_.DeQue<uint8_t>();
    gm2ubParams_.blockLen = length * sizeof(T1);
    DataCopyPad(yGM_[yOffsetInGM * sizeof(T1)], yTensor, gm2ubParams_);
    dataQue_.FreeTensor(yTensor);
}
__aicore__ inline int64_t Min(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

template <typename T1>
__aicore__ inline void Bicubic2dAADataCopy<T1>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    if (blockIdx_ < tailBlockNum_) {
        totalLength_ += 1;
        totalOffset_ = blockIdx_ * totalLength_;
    } else {
        totalOffset_ = blockIdx_ * totalLength_ + tailBlockNum_;
    }

    for (int64_t loop = 0; loop < totalLength_; loop += ubFactor_) {
        int64_t length = Min(ubFactor_, totalLength_ - loop);
        int64_t offset = totalOffset_ + loop;
        CopyIn(offset, length);
        CopyOut(offset, length);
    }
}
} // namespace UpsampleBicubic2dAA

#endif // UPSAMPLE_BICUBIC2D_AA_DATA_COPY_H