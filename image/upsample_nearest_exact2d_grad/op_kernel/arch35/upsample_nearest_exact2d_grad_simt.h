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
 * \file upsample_nearest_exact2d_grad_simt.h
 * \brief upsample_nearest_exact2d_grad_simt
 */

#ifndef UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT
#define UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest_exact2d_grad_tiling_data.h"
#include "./upsample_nearest_exact2d_grad_simt_base.h"
#include "simt_api/asc_simt.h"

namespace UpsampleNearestExact2dGrad {
using namespace AscendC;

template <typename T1, typename T2, bool isExact, uint64_t schId>
class NearestExact2dGradSimt {
public:
    __aicore__ inline NearestExact2dGradSimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UpsampleNearestExact2dGradRegBaseTilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    const UpsampleNearestExact2dGradRegBaseTilingData* tilingData;
    int32_t blockIdx = 0;
    GlobalTensor<T1> inputGm;
    GlobalTensor<T1> outputGm;
};

template <typename T1, typename T2, bool isExact, uint64_t schId>
__aicore__ inline void NearestExact2dGradSimt<T1, T2, isExact, schId>::Init(
    GM_ADDR x, GM_ADDR y, const UpsampleNearestExact2dGradRegBaseTilingData* __restrict tiling)
{
    inputGm.SetGlobalBuffer((__gm__ T1*)x);
    outputGm.SetGlobalBuffer((__gm__ T1*)y);
    blockIdx = GetBlockIdx();
    tilingData = tiling;
}

template <typename T1, typename T2, bool isExact, uint64_t schId>
__aicore__ inline void NearestExact2dGradSimt<T1, T2, isExact, schId>::Process()
{
    if (blockIdx >= static_cast<int32_t>(GetBlockNum())) {
        return;
    }
    T2 blkProcessNum = tilingData->blkProcessNum;
    T2 blkStartOffset = blockIdx * blkProcessNum;
    if (blockIdx < tilingData->tailBlockNum) {
        blkProcessNum = blkProcessNum + 1;
        blkStartOffset = blkStartOffset + blockIdx;
    } else {
        blkStartOffset = blkStartOffset + tilingData->tailBlockNum;
    }
    T2 mW = 0, shiftW = 0, mH = 0, shiftH = 0;
    T2 lenDstW = static_cast<T2>(tilingData->outW);
    T2 lenDstH = static_cast<T2>(tilingData->outH);
    T2 lenN = static_cast<T2>(tilingData->lenN);
    T2 lenC = static_cast<T2>(tilingData->lenC);
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    if constexpr (schId != SCH_ID_1) {
        GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    }
    T2 lenSrcW = static_cast<T2>(tilingData->inW);
    T2 lenSrcH = static_cast<T2>(tilingData->inH);
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        asc_vf_call<calleeInt64<T1, T2, isExact, schId>>(
            dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW,
            tilingData->scaleH, tilingData->scaleW);
    } else {
        asc_vf_call<calleeInt32<T1, T2, isExact, schId>>(
            dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            blkStartOffset, blkProcessNum, lenN, lenC, mH, shiftH, mW, shiftW, lenSrcH, lenSrcW, lenDstH, lenDstW,
            tilingData->scaleH, tilingData->scaleW);
    }
}
} // namespace UpsampleNearestExact2dGrad

#endif // UPSAMPLE_NEAREST_EXACT2D_GRAD_SIMT
