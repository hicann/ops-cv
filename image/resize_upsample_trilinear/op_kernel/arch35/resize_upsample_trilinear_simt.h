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
 * \file resize_upsample_trilinear_simt.h
 * \brief ResizeUpsampleTrilinear SIMT kernel implementation for arch35.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "./resize_upsample_trilinear_simt_base.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

template <typename T1, typename T2>
class ResizeUpsampleTrilinearSimt {
public:
    __aicore__ inline ResizeUpsampleTrilinearSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    const ResizeUpsampleTrilinearRegBaseTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::Init(
    GM_ADDR x, GM_ADDR y, const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData)
{
    inputGm_.SetGlobalBuffer((__gm__ T1*)x);
    outputGm_.SetGlobalBuffer((__gm__ T1*)y);
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
}

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::Process()
{
    if (blockIdx_ >= static_cast<int32_t>(GetBlockNum())) {
        return;
    }
    T2 blkProcessNum = static_cast<T2>(tilingData_->blkProcessNum);
    T2 blkStartOffset = static_cast<T2>(blockIdx_) * blkProcessNum;
    if (blockIdx_ < tilingData_->tailBlockNum) {
        blkProcessNum = blkProcessNum + 1;
        blkStartOffset = blkStartOffset + static_cast<T2>(blockIdx_);
    } else {
        blkStartOffset = blkStartOffset + static_cast<T2>(tilingData_->tailBlockNum);
    }

    T2 lenSrcD = static_cast<T2>(tilingData_->inD);
    T2 lenSrcH = static_cast<T2>(tilingData_->inH);
    T2 lenSrcW = static_cast<T2>(tilingData_->inW);
    T2 lenDstD = static_cast<T2>(tilingData_->outD);
    T2 lenDstH = static_cast<T2>(tilingData_->outH);
    T2 lenDstW = static_cast<T2>(tilingData_->outW);

    T2 mW = 0;
    T2 shiftW = 0;
    T2 mH = 0;
    T2 shiftH = 0;
    T2 mD = 0;
    T2 shiftD = 0;
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    GetUintDivMagicAndShift(mD, shiftD, lenDstD);

    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        asc_vf_call<calleeInt64<T1, T2>>(
            dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
            blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD,
            lenDstH, lenDstW, tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW, tilingData_->alignCorners);
    } else {
        asc_vf_call<calleeInt32<T1, T2>>(
            dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
            blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD,
            lenDstH, lenDstW, tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW, tilingData_->alignCorners);
    }
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_