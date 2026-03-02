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
 * \file upsample_nearest3d_grad_simt.h
 * \brief upsample_nearest3d_grad_simt
 */

#ifndef UPSAMPLE_NEAREST3D_GRAD_SIMT
#define UPSAMPLE_NEAREST3D_GRAD_SIMT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest3d_grad_tiling_data.h"
#include "./upsample_nearest3d_grad_simt_base.h"

namespace UpsampleNearest3dGrad {
using namespace AscendC;

template <typename T1, typename T2, bool isExtra, uint64_t schId>
class Nearest3dGradSimt {
public:
    __aicore__ inline Nearest3dGradSimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UpsampleNearest3dGradRegBaseTilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    const UpsampleNearest3dGradRegBaseTilingData* tilingData;
    int32_t blockIdx = 0;
    GlobalTensor<T1> inputGm;
    GlobalTensor<T1> outputGm;
};

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ inline void Nearest3dGradSimt<T1, T2, isExtra, schId>::Init(
    GM_ADDR x, GM_ADDR y, const UpsampleNearest3dGradRegBaseTilingData* __restrict tiling)
{
    inputGm.SetGlobalBuffer((__gm__ T1*)x);
    outputGm.SetGlobalBuffer((__gm__ T1*)y);
    blockIdx = GetBlockIdx();
    tilingData = tiling;
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ inline void Nearest3dGradSimt<T1, T2, isExtra, schId>::Process()
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
    T2 mW = 0, shiftW = 0, mH = 0, shiftH = 0, mD = 0, shiftD = 0;
    T2 lenDstW = static_cast<T2>(tilingData->outW);
    T2 lenDstH = static_cast<T2>(tilingData->outH);
    T2 lenDstD = static_cast<T2>(tilingData->outD);
    T2 lenN = static_cast<T2>(tilingData->lenN);
    T2 lenC = static_cast<T2>(tilingData->lenC);
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    if constexpr (schId != SCH_ID_1) {
        GetUintDivMagicAndShift(mD, shiftD, lenDstD);
    }
    T2 lenSrcW = static_cast<T2>(tilingData->inW);
    T2 lenSrcH = static_cast<T2>(tilingData->inH);
    T2 lenSrcD = static_cast<T2>(tilingData->inD);
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, T2, isExtra, schId>>(
            Simt::Dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW,
            lenDstD, lenDstH, lenDstW, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW);
    } else {
        Simt::VF_CALL<calleeInt32<T1, T2, isExtra, schId>>(
            Simt::Dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW,
            lenDstD, lenDstH, lenDstW, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW);
    }
}
} // namespace UpsampleNearest3dGrad

#endif // UPSAMPLE_NEAREST3D_GRAD_SIMT
