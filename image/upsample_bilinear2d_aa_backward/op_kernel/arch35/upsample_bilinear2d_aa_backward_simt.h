/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bilinear2d_aa_backward_simt.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_H
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_bilinear2d_aa_backward_tiling_data.h"
#include "./upsample_bilinear2d_aa_backward_simt_base.h"

namespace UpsampleBilinear2dAABackward {
using namespace AscendC;

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
class Bilinear2dAABackwardSimt {
public:
    __aicore__ inline Bilinear2dAABackwardSimt(){};

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, const UpsampleBilinear2dAABackwardRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearOut();
    const UpsampleBilinear2dAABackwardRegBaseTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
__aicore__ inline void Bilinear2dAABackwardSimt<T1, T2, T3, isDetermine>::Init(
    GM_ADDR input, GM_ADDR output, const UpsampleBilinear2dAABackwardRegBaseTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    inputGm_.SetGlobalBuffer((__gm__ T1*)input);
    outputGm_.SetGlobalBuffer((__gm__ T1*)output);
}

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
__aicore__ inline void Bilinear2dAABackwardSimt<T1, T2, T3, isDetermine>::ClearOut()
{
    T3 initBlkProcessNum = tilingData_->initBlkProcessNum;
    T3 initBlkStartOffset = blockIdx_ * initBlkProcessNum;
    if (blockIdx_ < tilingData_->initTailBlockNum) {
        initBlkProcessNum = initBlkProcessNum + 1;
        initBlkStartOffset = initBlkStartOffset + blockIdx_;
    } else {
        initBlkStartOffset = initBlkStartOffset + tilingData_->initTailBlockNum;
    }
    InitOutput<T1>(outputGm_[initBlkStartOffset], initBlkProcessNum, static_cast<T1>(0.0f));
}

template <typename T1, typename T2, typename T3, uint64_t isDetermine>
__aicore__ inline void Bilinear2dAABackwardSimt<T1, T2, T3, isDetermine>::Process()
{
    if constexpr (isDetermine == 0) {
        if (blockIdx_ < tilingData_->initRealCoreNum) {
            ClearOut();
        }
        SyncAll();
    }

    if (blockIdx_ < tilingData_->realCoreNum) {
        T3 blkProcessNum = tilingData_->blkProcessNum;
        T3 blkStartOffset = blockIdx_ * blkProcessNum;
        if (blockIdx_ < tilingData_->tailBlockNum) {
            blkProcessNum = blkProcessNum + 1;
            blkStartOffset = blkStartOffset + blockIdx_;
        } else {
            blkStartOffset = blkStartOffset + tilingData_->tailBlockNum;
        }
        T3 lenN = static_cast<T3>(tilingData_->lenN);
        T3 lenC = static_cast<T3>(tilingData_->lenC);
        T3 maxInterpSizeH = static_cast<T3>(tilingData_->maxInterpSizeH);
        T3 maxInterpSizeW = static_cast<T3>(tilingData_->maxInterpSizeW);
        T3 lenSrcW = static_cast<T3>(tilingData_->outW);
        T3 lenSrcH = static_cast<T3>(tilingData_->outH);
        T3 lenDstW = static_cast<T3>(tilingData_->inW);
        T3 lenDstH = static_cast<T3>(tilingData_->inH);
        if constexpr (isDetermine == 0) {
            lenSrcW = static_cast<T3>(tilingData_->inW);
            lenSrcH = static_cast<T3>(tilingData_->inH);
            lenDstW = static_cast<T3>(tilingData_->outW);
            lenDstH = static_cast<T3>(tilingData_->outH);
        }
        T2 mW = 0;
        T2 shiftW = 0;
        T2 mH = 0;
        T2 shiftH = 0;
        GetUintDivMagicAndShift(mW, shiftW, static_cast<T2>(lenDstW));
        GetUintDivMagicAndShift(mH, shiftH, static_cast<T2>(lenDstH));
        if constexpr (sizeof(T2) == sizeof(uint64_t)) {
            Simt::VF_CALL<calleeInt64<T1, T2, T3, isDetermine>>(
                Simt::Dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
                blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW,
                maxInterpSizeH, maxInterpSizeW, tilingData_->scaleH, tilingData_->scaleW, tilingData_->invScaleH,
                tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW);
        } else {
            Simt::VF_CALL<calleeInt32<T1, T2, T3, isDetermine>>(
                Simt::Dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
                blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW,
                maxInterpSizeH, maxInterpSizeW, tilingData_->scaleH, tilingData_->scaleW, tilingData_->invScaleH,
                tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW);
        }
    }
}
} // namespace UpsampleBilinear2dAABackward

#endif // UPSAMPLE_BILINEAR2D_AA_BACKWARD_SIMT_H
