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
 * \file resize_linear_grad_tensor_move.h
 * \brief resize_linear_grad_tensor_move
 */

#ifndef RESIZE_LINEAR_GRAD_TENSOR_MOVE_H_
#define RESIZE_LINEAR_GRAD_TENSOR_MOVE_H_

#include "kernel_operator.h"

namespace ResizeLinearGrad {
using namespace AscendC;

template <typename T>
class ResizeLinearGradTensorMove {
public:
    __aicore__ inline ResizeLinearGradTensorMove(TPipe &pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
        const ResizeLinearGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);

private:
    TPipe &pipe_;
    const ResizeLinearGradTilingData *tilingData_;
    constexpr static int32_t bufferDb = 2;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, bufferDb> dataQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockIdx_ = 0;
    DataCopyExtParams copyParams_{ 1, 0, 0, 0, 0 };
    DataCopyPadExtParams<T> padParams_{ false, 0, 0, 0 };
};

template <typename T>
__aicore__ inline void ResizeLinearGradTensorMove<T>::Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
    const ResizeLinearGradTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    int64_t blockOffset_ = blockIdx_ * tilingData_->blkProcessNum;
    xGm_.SetGlobalBuffer((__gm__ T *)(grads) + blockOffset_);
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + blockOffset_);
    pipe_.InitBuffer(dataQueue_, 2,
        tilingData_->lenSrcLOrUb); // 注意此时的lenSrcLOrUb其实是ubSize，因为节省不必要的tilingdata
}

template <typename T>
__aicore__ inline void ResizeLinearGradTensorMove<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    copyParams_.blockLen = dataLen * sizeof(T);
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, xGm_[offset], copyParams_, padParams_);
    dataQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ResizeLinearGradTensorMove<T>::CopyOut(int64_t offset, int64_t dataLen)
{
    copyParams_.blockLen = dataLen * sizeof(T);
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, copyParams_);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ResizeLinearGradTensorMove<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    int64_t loopSize = tilingData_->ubLoopSizeB;
    int64_t dataLen = tilingData_->ubFactorTailB;
    if (blockIdx_ == tilingData_->realCoreNum - 1) {
        loopSize = tilingData_->ubLoopSizeT;
        dataLen = tilingData_->ubFactorTailT;
    }
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        offset = idx * tilingData_->ubFactor;
        CopyIn(offset, tilingData_->ubFactor);
        CopyOut(offset, tilingData_->ubFactor);
    }
    offset = (loopSize - 1) * tilingData_->ubFactor;
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}
} // namespace ResizeLinearGrad

#endif // RESIZE_LINEAR_GRAD_TENSOR_MOVE_H_
