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
 * \file resize_bilinear_v2_all_copy.h
 * \brief resize_bilinear_v2_all_copy
 */
#ifndef RESIZE_BILINEAR_V2_ALL_COPY_H
#define RESIZE_BILINEAR_V2_ALL_COPY_H

#include "../inc/platform.h"
#include "kernel_operator.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

template <typename T_DATA>
class ResizeBilinearV2AllCopy : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2AllCopy(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2TilingData *data);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t xOffsetInGM, int64_t length);
    __aicore__ inline void CopyOut(int64_t yOffsetInGM, int64_t length);

    const ResizeBilinearV2TilingData *tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;

    int64_t totalLength_;
    int64_t totalOffset_;
    int64_t ubFactor_;
};

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2AllCopy<T_DATA>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2TilingData *data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    int64_t bufferSize = tilingData_->ubCFactor * sizeof(T_DATA);
    this->pipe_->InitBuffer(dataQue_, 2, bufferSize);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2AllCopy<T_DATA>::CopyIn(int64_t xOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> xTensor = dataQue_.AllocTensor<uint8_t>();

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = 1;
    gm2ubParams.blockLen = length * sizeof(T_DATA);
    gm2ubParams.srcStride = 0;
    gm2ubParams.dstStride = 0;

    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(xTensor, xGM_[xOffsetInGM * sizeof(T_DATA)], gm2ubParams, padParams);

    dataQue_.EnQue(xTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2AllCopy<T_DATA>::CopyOut(int64_t yOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> yTensor = dataQue_.DeQue<uint8_t>();

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = 1;
    ub2gmParams.blockLen = length * sizeof(T_DATA);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = 0;
    DataCopyPad(yGM_[yOffsetInGM * sizeof(T_DATA)], yTensor, ub2gmParams);

    dataQue_.FreeTensor(yTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2AllCopy<T_DATA>::Process()
{
    int64_t blockIdx = GetBlockIdx();

    if (blockIdx > tilingData_->realCoreNum) {
        return;
    }

    totalLength_ = tilingData_->splitBlockFactor;
    if (blockIdx < tilingData_->splitBlockTailFactor) {
        totalLength_ += 1;
        totalOffset_ = blockIdx * totalLength_;
    } else {
        totalOffset_ = blockIdx * totalLength_ + tilingData_->splitBlockTailFactor;
    }

    ubFactor_ = tilingData_->ubCFactor;

    for (int64_t loop = 0; loop < totalLength_; loop += ubFactor_) {
        int64_t length = this->Min(ubFactor_, totalLength_ - loop);
        int64_t offset = totalOffset_ + loop;

        CopyIn(offset, length);
        CopyOut(offset, length);
    }
}

}  // namespace ResizeBilinearV2

#endif  // RESIZE_BILINEAR_V2_ALL_COPY_H
