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
 * \file upsample_bicubic2d_aa_simt.h
 * \brief upsample_bicubic2d_aa_simt
 */

#ifndef UPSAMPLE_BICUBIC2D_AA_SIMT
#define UPSAMPLE_BICUBIC2D_AA_SIMT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bicubic2d_aa_tiling_data.h"
#include "./upsample_bicubic2d_aa_simt_base.h"

namespace UpsampleBicubic2dAA {
using namespace AscendC;

template <typename T1, typename T2, typename T3, uint64_t schId>
class Bicubic2dAASimt {
public:
    __aicore__ inline Bicubic2dAASimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UpsampleBicubic2dAARegBaseTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    const UpsampleBicubic2dAARegBaseTilingData *tilingData_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, typename T2, typename T3, uint64_t schId>
__aicore__ inline void Bicubic2dAASimt<T1, T2, T3, schId>::Init(GM_ADDR x, GM_ADDR y,
    const UpsampleBicubic2dAARegBaseTilingData *__restrict tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    inputGm_.SetGlobalBuffer((__gm__ T1 *)x);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
}

template <typename T1, typename T2, typename T3, uint64_t schId>
__aicore__ inline void Bicubic2dAASimt<T1, T2, T3, schId>::Process()
{
    if (blockIdx_ >= static_cast<int32_t>(GetBlockNum())) {
        return;
    }
    T3 blkProcessNum = tilingData_->blkProcessNum;
    T3 blkStartOffset = blockIdx_ * blkProcessNum;
    if (blockIdx_ < tilingData_->tailBlockNum) {
        blkProcessNum = blkProcessNum + 1;
        blkStartOffset = blkStartOffset + blockIdx_;
    } else {
        blkStartOffset = blkStartOffset + tilingData_->tailBlockNum;
    }
    T2 mW = 0;
    T2 shiftW = 0;
    T2 mH = 0;
    T2 shiftH = 0;
    T3 lenDstW = static_cast<T3>(tilingData_->outW);
    T3 lenDstH = static_cast<T3>(tilingData_->outH);
    T3 lenN = static_cast<T3>(tilingData_->lenN);
    T3 lenC = static_cast<T3>(tilingData_->lenC);
    GetUintDivMagicAndShift(mW, shiftW, static_cast<T2>(lenDstW));
    GetUintDivMagicAndShift(mH, shiftH, static_cast<T2>(lenDstH));
    T3 lenSrcW = static_cast<T3>(tilingData_->inW);
    T3 lenSrcH = static_cast<T3>(tilingData_->inH);
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, T2, T3, schId>>(Simt::Dim3(THREAD_NUM_B64),
            (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()), blkStartOffset, blkProcessNum, 
            lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, tilingData_->scaleH, tilingData_->scaleW, 
            tilingData_->invScaleH, tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW);
    } else {
        Simt::VF_CALL<calleeInt32<T1, T2, T3, schId>>(Simt::Dim3(THREAD_NUM_B32),
            (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()), blkStartOffset, blkProcessNum, 
            lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW, tilingData_->scaleH, tilingData_->scaleW, 
            tilingData_->invScaleH, tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW);
    }
}
} // namespace UpsampleBicubic2dAA

#endif // UPSAMPLE_BICUBIC2D_AA_SIMT