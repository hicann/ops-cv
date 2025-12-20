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
class TILING_KEY_DATA_COPY_NHWC_JH : public ResizeNearestNeighborV2Base {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_JH(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData);
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
    __aicore__ inline void ComputeJhCutHw0(
        jhParams &loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset);
    __aicore__ inline void ComputeJhCutHw2(
        jhParams &loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset);
    __aicore__ inline void ComputeSmallJhCut(processParams &params);
    __aicore__ inline void ComputeHwLoop1(int64_t no, int64_t nLoopOnce, processParams &params);
    __aicore__ inline void ComputeHwLoop2(int64_t no, int64_t nLoopOnce, processParams &params);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail, int64_t hwTimes, int64_t hwTail);

    constexpr static int64_t bufferNum = 2;

private:
    const ResizeNearestNeighborV2TilingData *tilingData_;
    TPipe pipe;
    int64_t blockIdx_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, bufferNum> xQue_;
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
};

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    bias_ = tilingData_->halfPixelCenters == 1 ? 0.5f : 0.0f;
    lenC = tilingData_->lenC;
    hScale_ = tilingData_->scaleH;
    wScale_ = tilingData_->scaleW;
    srcHSize_ = tilingData->lenSrcH;
    srcWSize_ = tilingData->lenSrcW;
    inputGm_.SetGlobalBuffer((__gm__ T *)x);
    outputGm_.SetGlobalBuffer((__gm__ T *)y);
    pipe.InitBuffer(xQue_, bufferNum, tilingData_->ubSize);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeJhCutHw0(
    jhParams &loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset)
{
    LocalTensor<T> inputUb = xQue_.AllocTensor<T>();
    for (int64_t howo = 0; howo < loopParams.hwOnceLoop; howo++) {
        int64_t ho = (hwBlockOffset + loopParams.hw * tilingData_->wcLoop + howo) / tilingData_->lenDesW;
        int64_t wo = (hwBlockOffset + loopParams.hw * tilingData_->wcLoop + howo) % tilingData_->lenDesW;
        int64_t h = this->Min(this->Floor(static_cast<float>((ho + bias_) * hScale_)), srcHSize_ - 1);
        int64_t w = this->Min(this->Floor(static_cast<float>((wo + bias_) * wScale_)), srcWSize_ - 1);
        int64_t inputOffset = srcNOffset + loopParams.no * tilingData_->lenCAlign + h * tilingData_->wcNum + w * lenC;
        copyParams.blockCount = loopParams.nLoopOnce;
        copyParams.blockLen = lenC * sizeof(T);
        copyParams.srcStride = (tilingData_->hwcNum - lenC) * sizeof(T);
        copyParams.dstStride = (loopParams.hwOnceLoop * lenC - lenC) * sizeof(T) / BIT32;
        DataCopyPad(inputUb[howo * lenC], inputGm_[inputOffset], copyParams, padParams);
    }
    xQue_.EnQue(inputUb);
    int64_t outputOffset = nBlockOffset + loopParams.no * tilingData_->alignCorners + hwBlockOffset * lenC +
                           loopParams.hw * tilingData_->wcLoop * tilingData_->lenC;
    LocalTensor<T> outputUb = xQue_.DeQue<T>();
    copyParams.blockCount = loopParams.nLoopOnce;
    copyParams.blockLen = loopParams.hwOnceLoop * lenC * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = (tilingData_->dstHwcNum - loopParams.hwOnceLoop * lenC) * sizeof(T);
    DataCopyPad(outputGm_[outputOffset], outputUb, copyParams);
    xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeJhCutHw2(
    jhParams &loopParams, int64_t srcNOffset, int64_t nBlockOffset, int64_t hwBlockOffset)
{
    LocalTensor<T> inputUb = xQue_.AllocTensor<T>();
    for (int64_t howo = 0; howo < loopParams.hwOnceLoop; howo++) {
        int64_t ho = (hwBlockOffset + loopParams.hw * tilingData_->wcLoop + howo) / tilingData_->lenDesW;
        int64_t wo = (hwBlockOffset + loopParams.hw * tilingData_->wcLoop + howo) % tilingData_->lenDesW;
        int64_t h = this->Min(this->Round(static_cast<float>(ho * hScale_)), srcHSize_ - 1);
        int64_t w = this->Min(this->Round(static_cast<float>(wo * wScale_)), srcWSize_ - 1);
        int64_t inputOffset = srcNOffset + loopParams.no * tilingData_->lenCAlign + h * tilingData_->wcNum + w * lenC;
        copyParams.blockCount = loopParams.nLoopOnce;
        copyParams.blockLen = lenC * sizeof(T);
        copyParams.srcStride = (tilingData_->hwcNum - lenC) * sizeof(T);
        copyParams.dstStride = (loopParams.hwOnceLoop * lenC - lenC) * sizeof(T) / BIT32;
        DataCopyPad(inputUb[howo * lenC], inputGm_[inputOffset], copyParams, padParams);
    }
    xQue_.EnQue(inputUb);
    int64_t outputOffset = nBlockOffset + loopParams.no * tilingData_->alignCorners + hwBlockOffset * lenC +
                           loopParams.hw * tilingData_->wcLoop * lenC;
    LocalTensor<T> outputUb = xQue_.DeQue<T>();
    copyParams.blockCount = loopParams.nLoopOnce;
    copyParams.blockLen = loopParams.hwOnceLoop * lenC * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = (tilingData_->dstHwcNum - loopParams.hwOnceLoop * lenC) * sizeof(T);
    DataCopyPad(outputGm_[outputOffset], outputUb, copyParams);
    xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeHwLoop1(
    int64_t no, int64_t nLoopOnce, processParams &params)
{
    jhParams loopParams;
    for (int64_t hw = 0; hw < params.hwLoopTimes; hw++) {
        loopParams.no = no;
        loopParams.nLoopOnce = nLoopOnce;
        loopParams.hw = hw;
        loopParams.hwOnceLoop = tilingData_->wcLoop;
        ComputeJhCutHw0(loopParams, params.srcNOffset, params.nOffset, params.hwOffset);
    }
    loopParams.no = no;
    loopParams.nLoopOnce = nLoopOnce;
    loopParams.hw = params.hwLoopTimes;
    loopParams.hwOnceLoop = params.hwLoopTail;
    ComputeJhCutHw0(loopParams, params.srcNOffset, params.nOffset, params.hwOffset);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeHwLoop2(
    int64_t no, int64_t nLoopOnce, processParams &params)
{
    jhParams loopParams;
    for (int64_t hw = 0; hw < params.hwLoopTimes; hw++) {
        loopParams.no = no;
        loopParams.nLoopOnce = nLoopOnce;
        loopParams.hw = hw;
        loopParams.hwOnceLoop = tilingData_->wcLoop;
        ComputeJhCutHw2(loopParams, params.srcNOffset, params.nOffset, params.hwOffset);
    }
    loopParams.no = no;
    loopParams.nLoopOnce = nLoopOnce;
    loopParams.hw = params.hwLoopTimes;
    loopParams.hwOnceLoop = params.hwLoopTail;
    ComputeJhCutHw2(loopParams, params.srcNOffset, params.nOffset, params.hwOffset);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::ComputeSmallJhCut(processParams &params)
{
    switch (tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < params.nLoopTimes; no++) {
                ComputeHwLoop1(no, tilingData_->nLoop, params);
            }
            ComputeHwLoop1(params.nLoopTimes, params.nLoopTail, params);
            break;
        }
        case 2: {
            for (int64_t no = 0; no < params.nLoopTimes; no++) {
                ComputeHwLoop2(no, tilingData_->nLoop, params);
            }
            ComputeHwLoop2(params.nLoopTimes, params.nLoopTail, params);
            break;
        }
        default:
            break;
    }
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
    switch (tilingData_->switchParams) {
        case 4:
            params.srcNOffset = blockIdx_ * tilingData_->splitBlockFactor * srcHSize_ * srcWSize_ * tilingData_->lenC;
            params.nOffset = blockIdx_ * tilingData_->lenN;
            params.hwOffset = 0;
            ComputeSmallJhCut(params);
            break;
        case 5:
            params.srcNOffset = 0;
            params.nOffset = 0;
            params.hwOffset = blockIdx_ * tilingData_->splitBlockFactor;
            ComputeSmallJhCut(params);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_JH<T>::Process()
{
    if (blockIdx_ == tilingData_->realCoreNum - 1) {
        ProcessPreCore(tilingData_->nLoopTimesLast,
            tilingData_->nLoopTailLast,
            tilingData_->wcLoopTimesLast,
            tilingData_->wcLoopTailLast);
    } else {
        ProcessPreCore(tilingData_->nLoopTimesBefore,
            tilingData_->splitBlockTailFactor,
            tilingData_->wcLoopTimesBefore,
            tilingData_->wcLoopTailBefore);
    }
}
}  // namespace ResizeNearestNeighborV2

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_JH_H
