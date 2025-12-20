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
 * \file resize_bicubic_v2_grad_all_copy.h
 * \brief resize_bicubic_v2_grad_all_copy
 */

#ifndef RESIZE_BICUBIC_V2_GRAD_ALL_COPY_H
#define RESIZE_BICUBIC_V2_GRAD_ALL_COPY_H

#include "resize_bicubic_v2_grad_base.h"

namespace ResizeBicubicV2Grad {
using namespace AscendC;

template <typename T_DATA>
class ResizeBicubicV2GradAllCopy {
public:
    __aicore__ inline ResizeBicubicV2GradAllCopy(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR y, TPipe *pipe, const ResizeBicubicV2GradAllCopyTilingData *tilingData);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t offset, int64_t length);
    __aicore__ inline void CopyOut(int64_t offset, int64_t length);

    TPipe *pipe_;
    const ResizeBicubicV2GradAllCopyTilingData *tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER_NUM> dataQue_;

    GlobalTensor<T_DATA> gradsGM_;
    GlobalTensor<T_DATA> yGM_;

    int64_t coreFactor_;
    int64_t coreOffset_;
};

template <typename T_DATA>
__aicore__ inline void ResizeBicubicV2GradAllCopy<T_DATA>::Init(
    GM_ADDR grads, GM_ADDR y, TPipe *pipe, const ResizeBicubicV2GradAllCopyTilingData *tilingData)
{
    pipe_ = pipe;
    tilingData_ = tilingData;

    gradsGM_.SetGlobalBuffer((__gm__ T_DATA *)grads);
    yGM_.SetGlobalBuffer((__gm__ T_DATA *)y);

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize();
    int64_t bufferSize = Ops::Base::CeilAlign<int64_t>(tilingData_->ubFactor * sizeof(T_DATA), ubBlockSize);
    pipe_->InitBuffer(dataQue_, DB_BUFFER_NUM, bufferSize);
}

template <typename T_DATA>
__aicore__ inline void ResizeBicubicV2GradAllCopy<T_DATA>::CopyIn(int64_t offset, int64_t length)
{
    LocalTensor<T_DATA> gradsTensor = dataQue_.AllocTensor<T_DATA>();

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = 1;
    gm2ubParams.blockLen = length * sizeof(T_DATA);
    gm2ubParams.srcStride = 0;
    gm2ubParams.dstStride = 0;

    DataCopyPadExtParams<T_DATA> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(gradsTensor, gradsGM_[offset], gm2ubParams, padParams);

    dataQue_.EnQue(gradsTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBicubicV2GradAllCopy<T_DATA>::CopyOut(int64_t offset, int64_t length)
{
    LocalTensor<T_DATA> yTensor = dataQue_.DeQue<T_DATA>();

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = 1;
    ub2gmParams.blockLen = length * sizeof(T_DATA);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = 0;

    DataCopyPad(yGM_[offset], yTensor, ub2gmParams);

    dataQue_.FreeTensor(yTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBicubicV2GradAllCopy<T_DATA>::Process()
{
    int64_t coreIdx = GetBlockIdx();
    if (coreIdx >= tilingData_->useCoreNum) {
        return;
    }

    coreFactor_ = tilingData_->coreFactor;
    coreOffset_ = coreIdx * tilingData_->coreFactor;
    if (coreIdx < tilingData_->coreTailFactor) {
        coreFactor_ += 1;
        coreOffset_ += coreIdx;
    } else {
        coreOffset_ += tilingData_->coreTailFactor;
    }

    for (int64_t ubLoop = 0; ubLoop < coreFactor_; ubLoop += tilingData_->ubFactor) {
        int64_t length = Min(tilingData_->ubFactor, coreFactor_ - ubLoop);
        int64_t offset = coreOffset_ + ubLoop;

        CopyIn(offset, length);
        CopyOut(offset, length);
    }
    return;
}

}  // namespace ResizeBicubicV2Grad

#endif  // RESIZE_BICUBIC_V2_GRAD_ALL_COPY_H
