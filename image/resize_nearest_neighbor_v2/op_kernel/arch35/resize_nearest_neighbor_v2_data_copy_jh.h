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
 * \file resize_nearest_neighbor_v2_data_copy_jh.h
 * \brief resize_nearest_neighbor_v2_data_copy_jh
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_JH_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_JH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_nearest_neighbor_v2_base.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

template <typename T>
class TILING_KEY_DATA_COPY_NHWC_JH : public ResizeNearestNeighborV2Base<T> {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_JH(){};
    __aicore__ inline void Process();
    struct processParams {
        int64_t nLoopTimes = 0;
        int64_t nLoopTail = 0;
        int64_t hwLoopTimes = 0;
        int64_t hwLoopTail = 0;
        int64_t srcNOffset = 0;
        int64_t nOffset = 0;
        int64_t hwOffset = 0;
    };
    struct jhParams {
        int64_t no = 0;
        int64_t nLoopOnce = 0;
        int64_t hw = 0;
        int64_t hwOnceLoop = 0;
    };

private:
    __aicore__ inline void ComputeJhCutHw(
        jhParams& loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset, int64_t mode);
    __aicore__ inline void ComputeSmallJhCut(processParams& params);
    __aicore__ inline void ComputeHwLoop(int64_t no, int64_t nLoopOnce, processParams& params, int64_t mode);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTimes, int64_t hwTail);
};

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeJhCutHw(
    jhParams& loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset, int64_t mode)
{
    LocalTensor<T> inputUb = this->xQue_.template AllocTensor<T>();
    for (int64_t howo = 0; howo < loopParams.hwOnceLoop; howo++) {
        int64_t ho = (hwBlockOffset + loopParams.hw * this->tilingData_->wcLoop + howo) / this->tilingData_->lenDesW;
        int64_t wo = (hwBlockOffset + loopParams.hw * this->tilingData_->wcLoop + howo) % this->tilingData_->lenDesW;
        int64_t h, w;
        if (mode == 0) {
            h = this->Min(this->Floor(static_cast<float>((ho + this->bias_) * this->hScale_)), this->srcHSize_ - 1);
            w = this->Min(this->Floor(static_cast<float>((wo + this->bias_) * this->wScale_)), this->srcWSize_ - 1);
        } else {
            h = this->Min(this->Round(static_cast<float>(ho * this->hScale_)), this->srcHSize_ - 1);
            w = this->Min(this->Round(static_cast<float>(wo * this->wScale_)), this->srcWSize_ - 1);
        }
        int64_t inputOffset =
            srcNOffset + loopParams.no * this->tilingData_->lenCAlign + h * this->tilingData_->wcNum + w * this->lenC_;
        this->copyParams_.blockCount = loopParams.nLoopOnce;
        this->copyParams_.blockLen = this->lenC_ * sizeof(T);
        this->copyParams_.srcStride = (this->tilingData_->hwcNum - this->lenC_) * sizeof(T);
        this->copyParams_.dstStride = (loopParams.hwOnceLoop * this->lenC_ - this->lenC_) * sizeof(T) / BIT32;
        DataCopyPad(inputUb[howo * this->lenC_], this->inputGm_[inputOffset], this->copyParams_, this->padParams_);
    }
    this->xQue_.EnQue(inputUb);
    int64_t outputOffset = nBlockOffset + loopParams.no * this->tilingData_->alignCorners +
                           hwBlockOffset * this->lenC_ + loopParams.hw * this->tilingData_->wcLoop * this->lenC_;
    LocalTensor<T> outputUb = this->xQue_.template DeQue<T>();
    this->copyParams_.blockCount = loopParams.nLoopOnce;
    this->copyParams_.blockLen = loopParams.hwOnceLoop * this->lenC_ * sizeof(T);
    this->copyParams_.srcStride = 0;
    this->copyParams_.dstStride = (this->tilingData_->dstHwcNum - loopParams.hwOnceLoop * this->lenC_) * sizeof(T);
    DataCopyPad(this->outputGm_[outputOffset], outputUb, this->copyParams_);
    this->xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeHwLoop(
    int64_t no, int64_t nLoopOnce, processParams& params, int64_t mode)
{
    jhParams loopParams;
    for (int64_t hw = 0; hw < params.hwLoopTimes; hw++) {
        loopParams.no = no;
        loopParams.nLoopOnce = nLoopOnce;
        loopParams.hw = hw;
        loopParams.hwOnceLoop = this->tilingData_->wcLoop;
        ComputeJhCutHw(loopParams, params.srcNOffset, params.nOffset, params.hwOffset, mode);
    }
    loopParams.no = no;
    loopParams.nLoopOnce = nLoopOnce;
    loopParams.hw = params.hwLoopTimes;
    loopParams.hwOnceLoop = params.hwLoopTail;
    ComputeJhCutHw(loopParams, params.srcNOffset, params.nOffset, params.hwOffset, mode);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeSmallJhCut(processParams &params)
{
    int64_t mode = this->tilingData_->condition;
    for (int64_t no = 0; no < params.nLoopTimes; no++) {
        ComputeHwLoop(no, this->tilingData_->nLoop, params, mode);
    }
    ComputeHwLoop(params.nLoopTimes, params.nLoopTail, params, mode);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ProcessPreCore(
    int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTimes, int64_t hwTail)
{
    processParams params;
    params.nLoopTimes = nLoopTimes;
    params.nLoopTail = nLoopTail;
    params.hwLoopTimes = hwTimes;
    params.hwLoopTail = hwTail;
    switch (this->tilingData_->switchParams) {
        case 4:
            params.srcNOffset =
                this->blockIdx_ * this->tilingData_->splitBlockFactor * this->srcHSize_ * this->srcWSize_ * this->lenC_;
            params.nOffset = this->blockIdx_ * this->tilingData_->lenN;
            params.hwOffset = 0;
            ComputeSmallJhCut(params);
            break;
        case 5:
            params.srcNOffset = 0;
            params.nOffset = 0;
            params.hwOffset = this->blockIdx_ * this->tilingData_->splitBlockFactor;
            ComputeSmallJhCut(params);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::Process()
{
    if (this->blockIdx_ == this->tilingData_->realCoreNum - 1) {
        ProcessPreCore(
            this->tilingData_->nLoopTimesLast, this->tilingData_->nLoopTailLast, this->tilingData_->wcLoopTimesLast,
            this->tilingData_->wcLoopTailLast);
    } else {
        ProcessPreCore(
            this->tilingData_->nLoopTimesBefore, this->tilingData_->splitBlockTailFactor,
            this->tilingData_->wcLoopTimesBefore, this->tilingData_->wcLoopTailBefore);
    }
}
} // namespace ResizeNearestNeighborV2

#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_JH_H