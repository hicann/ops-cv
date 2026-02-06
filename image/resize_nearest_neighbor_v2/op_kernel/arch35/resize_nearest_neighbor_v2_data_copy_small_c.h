/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2_data_copy_small_c.h
 * \brief resize_nearest_neighbor_v2_data_copy_small_c
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_nearest_neighbor_v2_base.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

template <typename T>
class TILING_KEY_DATA_COPY_NHWC_S_C : public ResizeNearestNeighborV2Base<T> {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_S_C(){};
    __aicore__ inline void Process();

private:
    __aicore__ inline void DataCopyInAndOut(int64_t nLoopOnce, int64_t inputOffset, int64_t outputOffset);
    __aicore__ inline void ComputeHw(int64_t no, int64_t nLoopOnce, int64_t mode);
    __aicore__ inline void ComputeSmallC(int64_t nLoopTimes, int64_t nLoopTail);
    __aicore__ inline void ComputeSmallCHw(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwOnceLoop);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTail);
    __aicore__ inline void ComputeHwOnce(int64_t no, int64_t nLoopOnce, int64_t hwLoop, int64_t mode);
};

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::DataCopyInAndOut(
    int64_t nLoopOnce, int64_t inputOffset, int64_t outputOffset)
{
    LocalTensor<T> inputUb = this->xQue_.template AllocTensor<T>();
    this->copyParams_.blockCount = nLoopOnce;
    this->copyParams_.blockLen = this->lenC_ * sizeof(T);
    this->copyParams_.srcStride = (this->tilingData_->hwcNum - this->lenC_) * sizeof(T);
    this->copyParams_.dstStride = 0;
    DataCopyPad(inputUb, this->inputGm_[inputOffset], this->copyParams_, this->padParams_);
    this->xQue_.EnQue(inputUb);
    LocalTensor<T> outputUb = this->xQue_.template DeQue<T>();
    this->copyParams_.srcStride = 0;
    this->copyParams_.dstStride = (this->tilingData_->dstHwcNum - this->lenC_) * sizeof(T);
    DataCopyPad(this->outputGm_[outputOffset], outputUb, this->copyParams_);
    this->xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHw(int64_t no, int64_t nLoopOnce, int64_t mode)
{
    for (int64_t ho = 0; ho < this->tilingData_->lenDesH; ho++) {
        for (int64_t wo = 0; wo < this->tilingData_->lenDesW; wo++) {
            int64_t h, w;
            if (mode == 0) {
                h = this->Min(this->Floor(static_cast<float>((ho + this->bias_) * this->hScale_)), this->srcHSize_ - 1);
                w = this->Min(this->Floor(static_cast<float>((wo + this->bias_) * this->wScale_)), this->srcWSize_ - 1);
            } else {
                h = this->Min(this->Round(float(ho) * this->hScale_), this->srcHSize_ - 1);
                w = this->Min(this->Round(float(wo) * this->wScale_), this->srcWSize_ - 1);
            }
            int64_t inputOffset = this->blockIdx_ * this->tilingData_->wcLoopTimesBefore +
                                  no * this->tilingData_->wcLoopTimesLast + h * this->tilingData_->wcNum +
                                  w * this->lenC_;
            int64_t outputOffset = this->blockIdx_ * this->tilingData_->wcLoop + no * this->tilingData_->lenCAlign +
                                   ho * this->tilingData_->dstWcNum + wo * this->lenC_;
            DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
        }
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeSmallC(int64_t nLoopTimes, int64_t nLoopTail)
{
    int64_t mode = this->tilingData_->condition;
    for (int64_t no = 0; no < nLoopTimes; no++) {
        ComputeHw(no, this->tilingData_->nLoop, mode);
    }
    ComputeHw(nLoopTimes, nLoopTail, mode);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHwOnce(
    int64_t no, int64_t nLoopOnce, int64_t hwLoop, int64_t mode)
{
    for (int64_t howo = 0; howo < hwLoop; howo++) {
        int64_t ho = (this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) / this->tilingData_->lenDesW;
        int64_t wo = (this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) % this->tilingData_->lenDesW;
        int64_t h, w;
        if (mode == 0) {
            h = this->Min(this->Floor(float(ho + this->bias_) * this->hScale_), this->srcHSize_ - 1);
            w = this->Min(this->Floor(float(wo + this->bias_) * this->wScale_), this->srcWSize_ - 1);
        } else {
            h = this->Min(this->Round(float(ho) * this->hScale_), this->srcHSize_ - 1);
            w = this->Min(this->Round(float(wo) * this->wScale_), this->srcWSize_ - 1);
        }
        int64_t inputOffset = no * this->tilingData_->wcLoopTimesLast + h * this->tilingData_->wcNum + w * this->lenC_;
        int64_t outputOffset = no * this->tilingData_->lenCAlign + ho * this->tilingData_->dstWcNum + wo * this->lenC_;
        DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeSmallCHw(
    int64_t nLoopTimes, int64_t nLoopTail, int64_t hwOnceLoop)
{
    int64_t mode = this->tilingData_->condition;
    for (int64_t no = 0; no < nLoopTimes; no++) {
        ComputeHwOnce(no, this->tilingData_->nLoop, hwOnceLoop, mode);
    }
    ComputeHwOnce(nLoopTimes, nLoopTail, hwOnceLoop, mode);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ProcessPreCore(
    int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTail)
{
    switch (this->tilingData_->switchParams) {
        case 0:
            ComputeSmallC(nLoopTimes, nLoopTail);
            break;
        case 2:
            ComputeSmallCHw(nLoopTimes, nLoopTail, hwTail);
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::Process()
{
    if (this->blockIdx_ == this->tilingData_->realCoreNum - 1) {
        ProcessPreCore(
            this->tilingData_->nLoopTimesLast, this->tilingData_->nLoopTailLast, this->tilingData_->wcLoopTailLast);
    } else {
        ProcessPreCore(
            this->tilingData_->nLoopTimesBefore, this->tilingData_->splitBlockTailFactor,
            this->tilingData_->wcLoopTailBefore);
    }
}
} // namespace ResizeNearestNeighborV2

#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H