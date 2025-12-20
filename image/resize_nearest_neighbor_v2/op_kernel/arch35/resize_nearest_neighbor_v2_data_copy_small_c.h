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
class TILING_KEY_DATA_COPY_NHWC_S_C : public ResizeNearestNeighborV2Base {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_S_C(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DataCopyInAndOut(int64_t nLoopOnce, int64_t inputOffset, int64_t outputOffset);
    __aicore__ inline void ComputeHw0(int64_t no, int64_t nLoopOnce);
    __aicore__ inline void ComputeHw2(int64_t no, int64_t nLoopOnce);
    __aicore__ inline void ComputeSmallC(int64_t nLoopTimes, int64_t nLoopTail);
    __aicore__ inline void ComputeSmallCHw(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwOnceLoop);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTail);
    __aicore__ inline void ComputeHwOnce0(int64_t no, int64_t nLoopOnce, int64_t hwLoop);
    __aicore__ inline void ComputeHwOnce2(int64_t no, int64_t nLoopOnce, int64_t hwLoop);

    constexpr static int64_t bufferNum = 2;

private:
    const ResizeNearestNeighborV2TilingData *tilingData_;
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, bufferNum> xQue_;
    int64_t blockIdx_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    bias_ = tilingData_->halfPixelCenters == 1 ? 0.5f : 0.0f;
    wScale_ = tilingData_->scaleW;
    hScale_ = tilingData_->scaleH;
    srcHSize_ = tilingData->lenSrcH;
    srcWSize_ = tilingData->lenSrcW;
    inputGm_.SetGlobalBuffer((__gm__ T *)x);
    outputGm_.SetGlobalBuffer((__gm__ T *)y);
    pipe.InitBuffer(xQue_, bufferNum, tilingData_->ubSize);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::DataCopyInAndOut(
    int64_t nLoopOnce, int64_t inputOffset, int64_t outputOffset)
{
    LocalTensor<T> inputUb = xQue_.AllocTensor<T>();
    copyParams.blockCount = nLoopOnce;
    copyParams.blockLen = tilingData_->lenC * sizeof(T);
    copyParams.srcStride = (tilingData_->hwcNum - tilingData_->lenC) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPad(inputUb, inputGm_[inputOffset], copyParams, padParams);
    xQue_.EnQue(inputUb);
    LocalTensor<T> outputUb = xQue_.DeQue<T>();
    copyParams.srcStride = 0;
    copyParams.dstStride = (tilingData_->dstHwcNum - tilingData_->lenC) * sizeof(T);
    DataCopyPad(outputGm_[outputOffset], outputUb, copyParams);
    xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHw0(int64_t no, int64_t nLoopOnce)
{
    for (int64_t ho = 0; ho < tilingData_->lenDesH; ho++) {
        for (int64_t wo = 0; wo < tilingData_->lenDesW; wo++) {
            int64_t h = this->Min(this->Floor(static_cast<float>((ho + bias_) * hScale_)), srcHSize_ - 1);
            int64_t w = this->Min(this->Floor(static_cast<float>((wo + bias_) * wScale_)), srcWSize_ - 1);
            int64_t inputOffset = blockIdx_ * tilingData_->wcLoopTimesBefore + no * tilingData_->wcLoopTimesLast +
                                  h * tilingData_->wcNum + w * tilingData_->lenC;
            int64_t outputOffset = blockIdx_ * tilingData_->wcLoop + no * tilingData_->lenCAlign +
                                   ho * tilingData_->dstWcNum + wo * tilingData_->lenC;
            DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
        }
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHw2(int64_t no, int64_t nLoopOnce)
{
    for (int64_t ho = 0; ho < tilingData_->lenDesH; ho++) {
        for (int64_t wo = 0; wo < tilingData_->lenDesW; wo++) {
            int64_t h = this->Min(this->Round(float(ho) * hScale_), srcHSize_ - 1);
            int64_t w = this->Min(this->Round(float(wo) * wScale_), srcWSize_ - 1);
            int64_t inputOffset = blockIdx_ * tilingData_->wcLoopTimesBefore + no * tilingData_->wcLoopTimesLast +
                                  h * tilingData_->wcNum + w * tilingData_->lenC;
            int64_t outputOffset = blockIdx_ * tilingData_->wcLoop + no * tilingData_->lenCAlign +
                                   ho * tilingData_->dstWcNum + wo * tilingData_->lenC;
            DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
        }
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeSmallC(int64_t nLoopTimes, int64_t nLoopTail)
{
    switch (tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < nLoopTimes; no++) {
                ComputeHw0(no, tilingData_->nLoop);
            }
            ComputeHw0(nLoopTimes, nLoopTail);
            break;
        }
        case 2: {
            for (int64_t no = 0; no < nLoopTimes; no++) {
                ComputeHw2(no, tilingData_->nLoop);
            }
            ComputeHw2(nLoopTimes, nLoopTail);
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHwOnce0(int64_t no, int64_t nLoopOnce, int64_t hwLoop)
{
    for (int64_t howo = 0; howo < hwLoop; howo++) {
        int64_t ho = (blockIdx_ * tilingData_->splitBlockFactor + howo) / tilingData_->lenDesW;
        int64_t wo = (blockIdx_ * tilingData_->splitBlockFactor + howo) % tilingData_->lenDesW;
        int64_t h = this->Min(this->Floor(float(ho + bias_) * hScale_), srcHSize_ - 1);
        int64_t w = this->Min(this->Floor(float(wo + bias_) * wScale_), srcWSize_ - 1);
        int64_t inputOffset = no * tilingData_->wcLoopTimesLast + h * tilingData_->wcNum + w * tilingData_->lenC;
        int64_t outputOffset = no * tilingData_->lenCAlign + ho * tilingData_->dstWcNum + wo * tilingData_->lenC;
        DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeHwOnce2(int64_t no, int64_t nLoopOnce, int64_t hwLoop)
{
    for (int64_t howo = 0; howo < hwLoop; howo++) {
        int64_t ho = (blockIdx_ * tilingData_->splitBlockFactor + howo) / tilingData_->lenDesW;
        int64_t wo = (blockIdx_ * tilingData_->splitBlockFactor + howo) % tilingData_->lenDesW;
        int64_t h = this->Min(this->Round(float(ho) * hScale_), srcHSize_ - 1);
        int64_t w = this->Min(this->Round(float(wo) * wScale_), srcWSize_ - 1);
        int64_t inputOffset = no * tilingData_->wcLoopTimesLast + h * tilingData_->wcNum + w * tilingData_->lenC;
        int64_t outputOffset = no * tilingData_->lenCAlign + ho * tilingData_->dstWcNum + wo * tilingData_->lenC;
        DataCopyInAndOut(nLoopOnce, inputOffset, outputOffset);
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ComputeSmallCHw(
    int64_t nLoopTimes, int64_t nLoopTail, int64_t hwOnceLoop)
{
    switch (tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < nLoopTimes; no++) {
                ComputeHwOnce0(no, tilingData_->nLoop, hwOnceLoop);
            }
            ComputeHwOnce0(nLoopTimes, nLoopTail, hwOnceLoop);
            break;
        }
        case 2: {
            for (int64_t no = 0; no < nLoopTimes; no++) {
                ComputeHwOnce2(no, tilingData_->nLoop, hwOnceLoop);
            }
            ComputeHwOnce2(nLoopTimes, nLoopTail, hwOnceLoop);
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_S_C<T>::ProcessPreCore(
    int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTail)
{
    switch (tilingData_->switchParams) {
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
    if (blockIdx_ == tilingData_->realCoreNum - 1) {
        ProcessPreCore(tilingData_->nLoopTimesLast, tilingData_->nLoopTailLast, tilingData_->wcLoopTailLast);
    } else {
        ProcessPreCore(tilingData_->nLoopTimesBefore, tilingData_->splitBlockTailFactor, tilingData_->wcLoopTailBefore);
    }
}
}  // namespace ResizeNearestNeighborV2

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H
