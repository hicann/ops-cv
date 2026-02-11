/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2_data_copy_nhwc.h
 * \brief resize_nearest_neighbor_v2_data_copy_nhwc.h
 */

#ifndef RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_NHWC_H
#define RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_NHWC_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::CreateAddrReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

template <typename T, int cutNH>
class ResizeNearestNeighborV2NHWC {
public:
    __aicore__ inline ResizeNearestNeighborV2NHWC(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeNearestNeighborV2TilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ComputeDataCopyUb(int64_t n, int64_t hLoopTimes, int64_t hTail);
    __aicore__ inline void DataCopyOut(LocalTensor<T> &yLocal, int64_t n, int64_t hoStart, int64_t oncehSize);
    __aicore__ inline void DataCopyIn(int64_t n, int64_t hiStart, int64_t hiSize, LocalTensor<T> &xLocal);
    __aicore__ inline void DataCopyOutNH(LocalTensor<T> &xLocal, int64_t oncehSize, int64_t idxStart);
    __aicore__ inline void DataCopyOutNorH(LocalTensor<T> &xLocal, int64_t oncehSize, int64_t hiStart, int64_t n);
    constexpr static int64_t bufferNum = 2;

private:
    const ResizeNearestNeighborV2TilingData *tilingData_;
    TPipe pipe;
    int64_t blockIdx_ = 0;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    DataCopyExtParams copyParams_{1, 1, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyParams repeatParams_{1, 1, 0, 0};

    uint32_t vfLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t ctimes_ = 0;
    int64_t lenC_ = 0;
    int64_t lenCAlign_ = 0;
    int64_t dstHwcNum_= 0;
    int64_t dstWSize_ = 0;
    int64_t srcWSize_ = 0;
    int64_t srcHSize_ = 0;
    int64_t hScale_ = 0;
    int64_t wScale_ = 0;

    int64_t realCoreNum_ = 0;
    int64_t hFactorSize_ = 0;

    int64_t wcNum_ = 0;
    int64_t hwcNum_= 0;
    
    int64_t condition_ = 0; // h 分核为1, n分核为0
    int64_t splitBlockFactor_ = 0;
    int64_t nLoopTimesBefore_ = 0;
    int64_t nLoopTimesLast_ = 0;
    int64_t dstWcNum_ = 0;
    
    int64_t hLoopTimesBB_ = 0;
    int64_t hLoopTimesBT_ = 0;
    int64_t hLoopTimesTB_ = 0;
    int64_t hLoopTimesTT_ = 0;
    int64_t dstWcAlign_ = 0;
    int64_t wcAlign_ = 0;
    int64_t xUb_ = 0;
    int64_t yUb_ = 0;
};

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeNearestNeighborV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T *)x);
    outputGm_.SetGlobalBuffer((__gm__ T *)y);

    realCoreNum_ = tilingData_->realCoreNum;
    lenC_ = tilingData_->lenC;
    lenCAlign_ = tilingData_->lenCAlign;
    dstHwcNum_ = tilingData->dstHwcNum;
    dstWSize_ = tilingData->lenDesW;
    srcWSize_ = tilingData->lenSrcW;
    srcHSize_ = tilingData->lenSrcH;
    hScale_ = static_cast<int64_t>(tilingData_->scaleH);
    wScale_ = static_cast<int64_t>(tilingData_->scaleW);
    hFactorSize_ = tilingData->splitFactorDesH; // h的ub切分值
    wcNum_ = tilingData_->wcNum;
    hwcNum_= tilingData_->hwcNum;
    dstWcAlign_ = dstWSize_ * lenCAlign_;
    wcAlign_ = srcWSize_ * lenCAlign_;
    condition_ = tilingData->condition;
    splitBlockFactor_ = tilingData->splitBlockFactor;
    nLoopTimesBefore_ = tilingData->nLoopTimesBefore;
    nLoopTimesLast_ = tilingData->nLoopTimesLast;
    dstWcNum_ = tilingData->dstWcNum;
    hLoopTimesBB_ = tilingData->wcLoopTimesBefore;
    hLoopTimesBT_ = tilingData->wcLoopTailBefore;
    hLoopTimesTB_ = tilingData->wcLoopTimesLast;
    hLoopTimesTT_ = tilingData->wcLoopTailLast;
    xUb_ = tilingData->splitFactorTailDesW;
    yUb_ = tilingData->splitFactorDesW;
    ctimes_ = CeilDivision(lenCAlign_, vfLen_);
    repeatParams_.blockCount = srcWSize_;
    repeatParams_.blockLen = (lenCAlign_ * sizeof(T)) / Ops::Base::GetUbBlockSize();
    repeatParams_.srcGap = 0;
    repeatParams_.dstGap = (lenCAlign_ * (wScale_ - 1) * sizeof(T)) / Ops::Base::GetUbBlockSize();
    pipe.InitBuffer(inQue_, bufferNum, xUb_);
    pipe.InitBuffer(outQue_, bufferNum, yUb_);
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::DataCopyIn(int64_t n, int64_t hiStart,
    int64_t hiSize, LocalTensor<T> &xLocal)
{
    int64_t srcOffset = 0;
    if (condition_ == 0) {
        srcOffset = (blockIdx_ *nLoopTimesBefore_ + n) * hwcNum_ + hiStart * wcNum_;
    } else {
        srcOffset = n * hwcNum_ + hiStart * wcNum_;
    }
    if (lenCAlign_ == lenC_) {
        copyParams_.blockCount = 1;
        copyParams_.blockLen = hiSize * wcNum_ * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
        AscendC::DataCopyPad(xLocal, inputGm_[srcOffset], copyParams_, padParams_);
    } else {
        LoopModeParams loopParams;
        copyParams_.blockCount = srcWSize_;
        copyParams_.blockLen = lenC_ * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = hiSize;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = wcNum_ * sizeof(T);
        loopParams.loop1DstStride = wcAlign_ * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(xLocal, inputGm_[srcOffset], copyParams_, padParams_);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::DataCopyOut(LocalTensor<T> &yLocal, int64_t n,
    int64_t hoStart, int64_t oncehSize)
{
    int64_t outOffset = 0;
    if (condition_ == 1) {
        outOffset = n * dstHwcNum_ + hoStart * dstWcNum_;
    } else {
        outOffset = (blockIdx_ *nLoopTimesBefore_ + n) * dstHwcNum_ + hoStart * dstWcNum_;
    }
    if (lenCAlign_ == lenC_) {
        copyParams_.blockCount = 1;
        copyParams_.blockLen = oncehSize * dstWcNum_ * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
        AscendC::DataCopyPad(outputGm_[outOffset], yLocal, copyParams_);
    } else {
        LoopModeParams loopParams;
        copyParams_.blockCount = dstWSize_;
        copyParams_.blockLen = lenC_ * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = oncehSize;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = dstWcAlign_ * sizeof(T);
        loopParams.loop1DstStride = dstWcNum_ * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(outputGm_[outOffset], yLocal, copyParams_);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::DataCopyOutNorH(LocalTensor<T> &xLocal,
    int64_t oncehSize, int64_t hiStart, int64_t n)
{
    for (int32_t h = 0; h < oncehSize; h++) {
        int64_t hoStart = (hiStart + h) * hScale_; 
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        // 做w方向的放大
        for (int64_t jj = 0; jj < wScale_; jj++) {
            AscendC::DataCopy(yLocal[jj*lenCAlign_], xLocal[h*wcAlign_], repeatParams_);
        }
        if constexpr(cutNH == 1) {
            // h反向的放大
            for (int64_t gg = 0; gg < (hScale_ - 1); gg++) {
                AscendC::DataCopy(yLocal[dstWcAlign_ + gg * dstWcAlign_], yLocal[0], dstWcAlign_);
            }
        }
        outQue_.EnQue<T>(yLocal);
        yLocal = outQue_.DeQue<T>();
        if constexpr(cutNH == 1) {
            DataCopyOut(yLocal, n, hoStart, hScale_); // 一个输入h对应的输出h ub可以一次放的下
            outQue_.FreeTensor(yLocal);
        } 
        if constexpr(cutNH == 0) {
            for (int64_t ii = 0; ii < hScale_; ii++) {
                DataCopyOut(yLocal, n, hoStart+ii, 1); // 一个输入h对应的输出h ub一次放不下，要分多次搬出
            }
            outQue_.FreeTensor(yLocal);
        }
    }
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::DataCopyOutNH(LocalTensor<T> &xLocal,
    int64_t oncehSize, int64_t idxStart)
{
    for (int32_t h = 0; h < oncehSize; h++) {
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        // 做w方向的放大
        for (int64_t jj = 0; jj < wScale_; jj++) {
            AscendC::DataCopy(yLocal[jj*lenCAlign_], xLocal[h*wcAlign_], repeatParams_);
        }
        outQue_.EnQue<T>(yLocal);
        yLocal = outQue_.DeQue<T>();
        int64_t offset = idxStart + h;
        int64_t ni = offset / srcHSize_;
        int64_t hiIn = offset % srcHSize_;
        int64_t hoStart = hiIn * hScale_;

        for (int64_t ii = 0; ii < hScale_; ii++) {
            DataCopyOut(yLocal, ni, hoStart+ii, 1); // 一个输入h对应的输出h ub一次放不下，要分多次搬出
        }
        outQue_.FreeTensor(yLocal);
    }
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::ComputeDataCopyUb(int64_t n,
    int64_t hLoopTimes, int64_t hTail)
{
    for (int64_t hi = 0; hi < hLoopTimes; hi++) { // 按照输入切分
        int64_t oncehSize = hi == hLoopTimes - 1 ? hTail : hFactorSize_;
        int64_t hiStart = 0;
        int64_t niStart = 0;
        int64_t idxStart = 0;
        if constexpr(cutNH == 2) {
            idxStart = blockIdx_ * splitBlockFactor_ + hi * hFactorSize_;
            niStart = idxStart / srcHSize_;
            hiStart = idxStart % srcHSize_;
        } else {
            hiStart = blockIdx_ * splitBlockFactor_ * condition_ + hi * hFactorSize_;
        }
        LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
        if constexpr(cutNH == 2) {
            DataCopyIn(niStart, hiStart, oncehSize, xLocal);
        } else {
            DataCopyIn(n, hiStart, oncehSize, xLocal);
        }
        inQue_.EnQue<T>(xLocal);
        xLocal = inQue_.DeQue<T>();
        if constexpr(cutNH == 2) {
            DataCopyOutNH(xLocal, oncehSize, idxStart);
        } else {
            DataCopyOutNorH(xLocal, oncehSize, hiStart, n);
        }
        inQue_.FreeTensor(xLocal);
    }
}

template <typename T, int cutNH>
__aicore__ inline void ResizeNearestNeighborV2NHWC<T, cutNH>::Process()
{
    // 兩种分核方式 n 和 h,谁可以分满核就分谁,，优先分n
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    if constexpr(cutNH == 2) {
        // nh合轴分核
        if (blockIdx_ < realCoreNum_ - 1) {
            ComputeDataCopyUb(0, hLoopTimesBB_, hLoopTimesBT_);

        } else {
            ComputeDataCopyUb(0, hLoopTimesTB_, hLoopTimesTT_);
        }
    } else {
        // n分核或者h分核
        if (blockIdx_ < realCoreNum_ - 1) {
            for (int64_t n = 0; n < nLoopTimesBefore_; n++) { // 前面的核处理多少个n，h分核时直接等于N
                ComputeDataCopyUb(n, hLoopTimesBB_, hLoopTimesBT_); // h做ub切分，hLoopTimesBB_是h方向循环次数，hLoopTimesBT_是尾快处理多少个h
            }
        } else {
            for (int64_t n = 0; n < nLoopTimesLast_; n++) { // 尾核处理多少个n，h分核时直接等于N
                ComputeDataCopyUb(n, hLoopTimesTB_, hLoopTimesTT_);
            }
        }
    }
}
}  // namespace ResizeNearestNeighborV2

#endif  // RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_NHWC_H
