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
 * \file resize_nearest_neighbor_v2_data_copy_big_c.h
 * \brief resize_nearest_neighbor_v2_data_copy_big_c
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_BIG_C_H
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_BIG_C_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "resize_nearest_neighbor_v2_base.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

template <typename T>
class TILING_KEY_DATA_COPY_NHWC_BIG_C : public ResizeNearestNeighborV2Base {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_BIG_C(){};

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeOnceBigC(int64_t onceC, int64_t inputOffset, int64_t outputOffset);
    __aicore__ inline void ComputeC(int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w);
    __aicore__ inline void ComputeBigC(int64_t nLoopTimes);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail);
    __aicore__ inline void ComputeBigCHw(int64_t lenN, int64_t hwOnceLoop);
    __aicore__ inline void ComputeLoopBigCHw(int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w);

    constexpr static int64_t bufferNum = 2;

private:
    const ResizeNearestNeighborV2TilingData *tilingData_;
    TPipe pipe;
    int64_t blockIdx_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, bufferNum> xQue_;

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::Init(
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
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ComputeOnceBigC(
    int64_t onceC, int64_t inputOffset, int64_t outputOffset)
{
    LocalTensor<T> inputUb = xQue_.AllocTensor<T>();
    copyParams.blockCount = 1;
    copyParams.blockLen = onceC * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(inputUb, inputGm_[inputOffset], copyParams, padParams);
    xQue_.EnQue(inputUb);
    LocalTensor<T> outputUb = xQue_.DeQue<T>();
    DataCopyPad(outputGm_[outputOffset], outputUb, copyParams);
    xQue_.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ComputeC(
    int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w)
{
    for (int64_t ci = 0; ci < tilingData_->wcLoopTimesBefore; ci++) {
        int64_t inputOffset = blockIdx_ * tilingData_->wcLoopTimesLast + no * tilingData_->hwcNum +
                              h * tilingData_->wcNum + w * tilingData_->lenC + ci * tilingData_->wcLoop;
        int64_t outputOffset = blockIdx_ * tilingData_->wcLoopTailLast + no * tilingData_->dstHwcNum +
                               ho * tilingData_->dstWcNum + wo * tilingData_->lenC + ci * tilingData_->wcLoop;
        ComputeOnceBigC(tilingData_->wcLoop, inputOffset, outputOffset);
    }
    int64_t inputOffset = blockIdx_ * tilingData_->wcLoopTimesLast + no * tilingData_->hwcNum + h * tilingData_->wcNum +
                          w * tilingData_->lenC + tilingData_->wcLoopTimesBefore * tilingData_->wcLoop;
    int64_t outputOffset = blockIdx_ * tilingData_->wcLoopTailLast + no * tilingData_->dstHwcNum +
                           ho * tilingData_->dstWcNum + wo * tilingData_->lenC +
                           tilingData_->wcLoopTimesBefore * tilingData_->wcLoop;
    ComputeOnceBigC(tilingData_->wcLoopTailBefore, inputOffset, outputOffset);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ComputeBigC(int64_t lenN)
{
    switch (tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < tilingData_->lenN; no++) {
                for (int64_t ho = 0; ho < tilingData_->lenDesH; ho++) {
                    for (int64_t wo = 0; wo < tilingData_->lenDesW; wo++) {
                        int64_t h = this->Min(this->Floor(static_cast<float>((ho + bias_) * hScale_)), srcHSize_ - 1);
                        int64_t w = this->Min(this->Floor(static_cast<float>((wo + bias_) * wScale_)), srcWSize_ - 1);
                        ComputeC(no, ho, wo, h, w);
                    }
                }
            }
            break;
        }
        case 2: {
            for (int64_t no = 0; no < tilingData_->lenN; no++) {
                for (int64_t ho = 0; ho < tilingData_->lenDesH; ho++) {
                    for (int64_t wo = 0; wo < tilingData_->lenDesW; wo++) {
                        int64_t h = this->Min(this->Round(static_cast<float>(ho * hScale_)), srcHSize_ - 1);
                        int64_t w = this->Min(this->Round(static_cast<float>(wo * wScale_)), srcWSize_ - 1);
                        ComputeC(no, ho, wo, h, w);
                    }
                }
            }
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ComputeLoopBigCHw(
    int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w)
{
    for (int64_t ci = 0; ci < tilingData_->wcLoopTimesBefore; ci++) {
        int64_t inputOffset =
            no * tilingData_->hwcNum + h * tilingData_->wcNum + w * tilingData_->lenC + ci * tilingData_->wcLoop;
        int64_t outputOffset = no * tilingData_->dstHwcNum + ho * tilingData_->dstWcNum + wo * tilingData_->lenC +
                               ci * tilingData_->wcLoop;
        ComputeOnceBigC(tilingData_->wcLoop, inputOffset, outputOffset);
    }
    int64_t inputOffset = no * tilingData_->hwcNum + h * tilingData_->wcNum + w * tilingData_->lenC +
                          tilingData_->wcLoopTimesBefore * tilingData_->wcLoop;
    int64_t outputOffset = no * tilingData_->dstHwcNum + ho * tilingData_->dstWcNum + wo * tilingData_->lenC +
                           tilingData_->wcLoopTimesBefore * tilingData_->wcLoop;
    ComputeOnceBigC(tilingData_->wcLoopTailBefore, inputOffset, outputOffset);
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ComputeBigCHw(int64_t lenN, int64_t hwOnceLoop)
{
    switch (tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < lenN; no++) {
                for (int64_t howo = 0; howo < hwOnceLoop; howo++) {
                    int64_t ho = (blockIdx_ * tilingData_->splitBlockFactor + howo) / tilingData_->lenDesW;
                    int64_t wo = (blockIdx_ * tilingData_->splitBlockFactor + howo) % tilingData_->lenDesW;
                    int64_t h = this->Min(this->Floor(static_cast<float>(ho * hScale_)), srcHSize_ - 1);
                    int64_t w = this->Min(this->Floor(static_cast<float>(wo * wScale_)), srcWSize_ - 1);
                    ComputeLoopBigCHw(no, ho, wo, h, w);
                }
            }
            break;
        }
        case 2: {
            for (int64_t no = 0; no < lenN; no++) {
                for (int64_t howo = 0; howo < hwOnceLoop; howo++) {
                    int64_t ho = (blockIdx_ * tilingData_->splitBlockFactor + howo) / tilingData_->lenDesW;
                    int64_t wo = (blockIdx_ * tilingData_->splitBlockFactor + howo) % tilingData_->lenDesW;
                    int64_t h = this->Min(this->Round(float(ho) * hScale_), srcHSize_ - 1);
                    int64_t w = this->Min(this->Round(float(wo) * wScale_), srcWSize_ - 1);
                    ComputeLoopBigCHw(no, ho, wo, h, w);
                }
            }
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail)
{
    switch (tilingData_->switchParams) {
        case 1:
            ComputeBigC(nLoopTimes);
            break;
        case 3:
            ComputeBigCHw(nLoopTimes, nLoopTail);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T>::Process()
{
    if (blockIdx_ == tilingData_->realCoreNum - 1) {
        ProcessPreCore(tilingData_->nLoopTimesLast, tilingData_->nLoopTailLast);
    } else {
        ProcessPreCore(tilingData_->nLoopTimesBefore, tilingData_->splitBlockTailFactor);
    }
}
}  // namespace ResizeNearestNeighborV2

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_BIG_C_H
