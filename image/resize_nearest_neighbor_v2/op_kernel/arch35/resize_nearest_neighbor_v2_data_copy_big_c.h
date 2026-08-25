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

// IdxType 决定输出位置 scalar 计算使用的类型：idxInt32 为 1 时为 int32_t，为 0 时为 int64_t。
template <typename T, typename IdxType = int64_t>
class TILING_KEY_DATA_COPY_NHWC_BIG_C : public ResizeNearestNeighborV2Base<T> {
public:
    __aicore__ inline TILING_KEY_DATA_COPY_NHWC_BIG_C(){};
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeOnceBigC(int64_t onceC, int64_t inputOffset, int64_t outputOffset);
    __aicore__ inline void ComputeC(int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w);
    __aicore__ inline void ComputeBigC(int64_t nLoopTimes);
    __aicore__ inline void ProcessPreCore(int64_t nLoopTimes, int64_t nLoopTail);
    __aicore__ inline void ComputeBigCHw(int64_t lenN, int64_t hwOnceLoop);
    __aicore__ inline void ComputeLoopBigCHw(int64_t no, int64_t ho, int64_t wo, int64_t h, int64_t w);
};

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ComputeOnceBigC(int64_t onceC, int64_t inputOffset,
                                                                                    int64_t outputOffset)
{
    LocalTensor<T> inputUb = this->xQue_.template AllocTensor<T>();
    this->copyParams_.blockCount = 1;
    this->copyParams_.blockLen = onceC * sizeof(T);
    this->copyParams_.srcStride = 0;
    this->copyParams_.dstStride = 0;

    DataCopyPad(inputUb, this->inputGm_[inputOffset], this->copyParams_, this->padParams_);
    this->xQue_.EnQue(inputUb);
    LocalTensor<T> outputUb = this->xQue_.template DeQue<T>();
    DataCopyPad(this->outputGm_[outputOffset], outputUb, this->copyParams_);
    this->xQue_.FreeTensor(outputUb);
}

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ComputeC(int64_t no, int64_t ho, int64_t wo,
                                                                             int64_t h, int64_t w)
{
    for (int64_t ci = 0; ci < this->tilingData_->wcLoopTimesBefore; ci++) {
        int64_t inputOffset = this->blockIdx_ * this->tilingData_->wcLoopTimesLast + no * this->tilingData_->hwcNum +
                              h * this->tilingData_->wcNum + w * this->lenC_ + ci * this->tilingData_->wcLoop;
        int64_t outputOffset = this->blockIdx_ * this->tilingData_->wcLoopTailLast + no * this->tilingData_->dstHwcNum +
                               ho * this->tilingData_->dstWcNum + wo * this->lenC_ + ci * this->tilingData_->wcLoop;
        ComputeOnceBigC(this->tilingData_->wcLoop, inputOffset, outputOffset);
    }
    int64_t inputOffset = this->blockIdx_ * this->tilingData_->wcLoopTimesLast + no * this->tilingData_->hwcNum +
                          h * this->tilingData_->wcNum + w * this->lenC_ +
                          this->tilingData_->wcLoopTimesBefore * this->tilingData_->wcLoop;
    int64_t outputOffset = this->blockIdx_ * this->tilingData_->wcLoopTailLast + no * this->tilingData_->dstHwcNum +
                           ho * this->tilingData_->dstWcNum + wo * this->lenC_ +
                           this->tilingData_->wcLoopTimesBefore * this->tilingData_->wcLoop;
    ComputeOnceBigC(this->tilingData_->wcLoopTailBefore, inputOffset, outputOffset);
}

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ComputeBigC(int64_t lenN)
{
    switch (this->tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < this->tilingData_->lenN; no++) {
                for (IdxType ho = 0; ho < static_cast<IdxType>(this->tilingData_->lenDesH); ho++) {
                    for (IdxType wo = 0; wo < static_cast<IdxType>(this->tilingData_->lenDesW); wo++) {
                        IdxType h = this->template MinT<IdxType>(
                            this->template FloorT<IdxType>(static_cast<float>((ho + this->bias_) * this->hScale_)),
                            static_cast<IdxType>(this->srcHSize_ - 1));
                        IdxType w = this->template MinT<IdxType>(
                            this->template FloorT<IdxType>(static_cast<float>((wo + this->bias_) * this->wScale_)),
                            static_cast<IdxType>(this->srcWSize_ - 1));
                        ComputeC(no, ho, wo, h, w);
                    }
                }
            }
            break;
        }
        case 2: {
            for (int64_t no = 0; no < this->tilingData_->lenN; no++) {
                for (IdxType ho = 0; ho < static_cast<IdxType>(this->tilingData_->lenDesH); ho++) {
                    for (IdxType wo = 0; wo < static_cast<IdxType>(this->tilingData_->lenDesW); wo++) {
                        IdxType h = this->template MinT<IdxType>(
                            this->template RoundT<IdxType>(static_cast<float>(ho * this->hScale_)),
                            static_cast<IdxType>(this->srcHSize_ - 1));
                        IdxType w = this->template MinT<IdxType>(
                            this->template RoundT<IdxType>(static_cast<float>(wo * this->wScale_)),
                            static_cast<IdxType>(this->srcWSize_ - 1));
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

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ComputeLoopBigCHw(int64_t no, int64_t ho,
                                                                                      int64_t wo, int64_t h, int64_t w)
{
    for (int64_t ci = 0; ci < this->tilingData_->wcLoopTimesBefore; ci++) {
        int64_t inputOffset = no * this->tilingData_->hwcNum + h * this->tilingData_->wcNum + w * this->lenC_ +
                              ci * this->tilingData_->wcLoop;
        int64_t outputOffset = no * this->tilingData_->dstHwcNum + ho * this->tilingData_->dstWcNum + wo * this->lenC_ +
                               ci * this->tilingData_->wcLoop;
        ComputeOnceBigC(this->tilingData_->wcLoop, inputOffset, outputOffset);
    }
    int64_t inputOffset = no * this->tilingData_->hwcNum + h * this->tilingData_->wcNum + w * this->lenC_ +
                          this->tilingData_->wcLoopTimesBefore * this->tilingData_->wcLoop;
    int64_t outputOffset = no * this->tilingData_->dstHwcNum + ho * this->tilingData_->dstWcNum + wo * this->lenC_ +
                           this->tilingData_->wcLoopTimesBefore * this->tilingData_->wcLoop;
    ComputeOnceBigC(this->tilingData_->wcLoopTailBefore, inputOffset, outputOffset);
}

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ComputeBigCHw(int64_t lenN, int64_t hwOnceLoop)
{
    switch (this->tilingData_->condition) {
        case 0: {
            for (int64_t no = 0; no < lenN; no++) {
                for (IdxType howo = 0; howo < static_cast<IdxType>(hwOnceLoop); howo++) {
                    IdxType ho = static_cast<IdxType>((this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) /
                                                      this->tilingData_->lenDesW);
                    IdxType wo = static_cast<IdxType>((this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) %
                                                      this->tilingData_->lenDesW);
                    IdxType h = this->template MinT<IdxType>(
                        this->template FloorT<IdxType>(static_cast<float>(ho * this->hScale_)),
                        static_cast<IdxType>(this->srcHSize_ - 1));
                    IdxType w = this->template MinT<IdxType>(
                        this->template FloorT<IdxType>(static_cast<float>(wo * this->wScale_)),
                        static_cast<IdxType>(this->srcWSize_ - 1));
                    ComputeLoopBigCHw(no, ho, wo, h, w);
                }
            }
            break;
        }
        case 2: {
            for (int64_t no = 0; no < lenN; no++) {
                for (IdxType howo = 0; howo < static_cast<IdxType>(hwOnceLoop); howo++) {
                    IdxType ho = static_cast<IdxType>((this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) /
                                                      this->tilingData_->lenDesW);
                    IdxType wo = static_cast<IdxType>((this->blockIdx_ * this->tilingData_->splitBlockFactor + howo) %
                                                      this->tilingData_->lenDesW);
                    IdxType h = this->template MinT<IdxType>(this->template RoundT<IdxType>(float(ho) * this->hScale_),
                                                             static_cast<IdxType>(this->srcHSize_ - 1));
                    IdxType w = this->template MinT<IdxType>(this->template RoundT<IdxType>(float(wo) * this->wScale_),
                                                             static_cast<IdxType>(this->srcWSize_ - 1));
                    ComputeLoopBigCHw(no, ho, wo, h, w);
                }
            }
            break;
        }
        default:
            break;
    }
}

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::ProcessPreCore(int64_t nLoopTimes,
                                                                                   int64_t nLoopTail)
{
    switch (this->tilingData_->switchParams) {
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

template <typename T, typename IdxType>
__aicore__ inline void TILING_KEY_DATA_COPY_NHWC_BIG_C<T, IdxType>::Process()
{
    if (this->blockIdx_ == this->tilingData_->realCoreNum - 1) {
        ProcessPreCore(this->tilingData_->nLoopTimesLast, this->tilingData_->nLoopTailLast);
    } else {
        ProcessPreCore(this->tilingData_->nLoopTimesBefore, this->tilingData_->splitBlockTailFactor);
    }
}
} // namespace ResizeNearestNeighborV2

#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_BIG_C_H
