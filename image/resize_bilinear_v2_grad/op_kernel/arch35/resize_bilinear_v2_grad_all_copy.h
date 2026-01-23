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
 * \file resize_bilinear_v2_grad_all_copy.h
 * \brief resize_bilinear_v2_grad_all_copy
 */
#ifndef RESIZE_BILINEAR_V2_GRAD_ALL_COPY_H
#define RESIZE_BILINEAR_V2_GRAD_ALL_COPY_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "resize_bilinear_v2_grad_base.h"

namespace ResizeBilinearV2Grad {
using namespace AscendC;

template <typename T_GRADS, typename T_OUT>
class ResizeBilinearV2GradAllCopy : public ResizeBilinearV2GradBase {
public:
    __aicore__ inline ResizeBilinearV2GradAllCopy(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2GradTilingData* data);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t gradsOffsetInGM, int64_t length);
    __aicore__ inline void CopyOut(int64_t yOffsetInGM, int64_t length);

    const ResizeBilinearV2GradTilingData* tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;

    int64_t totalLength_;
    int64_t totalOffset_;
};

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradAllCopy<T_GRADS, T_OUT>::Init(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2GradTilingData* data)
{
    this->BaseInit(grads, y, pipe);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize();
    tilingData_ = data;
    int64_t bufferSize = Ops::Base::CeilAlign<int64_t>(tilingData_->ubCFactor * sizeof(T_GRADS), ubBlockSize);
    this->pipe_->InitBuffer(dataQue_, 2, bufferSize);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradAllCopy<T_GRADS, T_OUT>::CopyIn(int64_t gradsOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> gradsTensor = dataQue_.AllocTensor<uint8_t>();

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = 1;
    gm2ubParams.blockLen = length * sizeof(T_GRADS);
    gm2ubParams.srcStride = 0;
    gm2ubParams.dstStride = 0;

    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(gradsTensor, gradsGM_[gradsOffsetInGM * sizeof(T_GRADS)], gm2ubParams, padParams);

    dataQue_.EnQue(gradsTensor);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradAllCopy<T_GRADS, T_OUT>::CopyOut(int64_t yOffsetInGM, int64_t length)
{
    LocalTensor<uint8_t> yTensor = dataQue_.DeQue<uint8_t>();

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = 1;
    ub2gmParams.blockLen = length * sizeof(T_OUT);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = 0;
    DataCopyPad(yGM_[yOffsetInGM * sizeof(T_OUT)], yTensor, ub2gmParams);

    dataQue_.FreeTensor(yTensor);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradAllCopy<T_GRADS, T_OUT>::Process()
{
    int64_t blockIdx = GetBlockIdx();

    if (blockIdx > tilingData_->realCoreNum) {
        return;
    }
    if (blockIdx < tilingData_->splitBlockTailFactor) {
        totalLength_ = tilingData_->splitBlockFactor + 1;
        totalOffset_ = blockIdx * totalLength_;
    } else {
        totalLength_ = tilingData_->splitBlockFactor;
        totalOffset_ = tilingData_->splitBlockFactor * blockIdx + tilingData_->splitBlockTailFactor;
    }

    for (int64_t loop = 0; loop < totalLength_; loop += tilingData_->ubCFactor) {
        int64_t length = this->Min(tilingData_->ubCFactor, totalLength_ - loop);
        int64_t offset = totalOffset_ + loop;

        CopyIn(offset, length);
        CopyOut(offset, length);
    }
    return;
}

} // namespace ResizeBilinearV2Grad

#endif // RESIZE_BILINEAR_V2_GRAD_ALL_COPY_H
