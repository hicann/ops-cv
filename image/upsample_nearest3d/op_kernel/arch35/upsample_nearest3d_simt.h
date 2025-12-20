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
 * \file upsample_nearest3d_simt.h
 * \brief upsample_nearest3d_simt
 */

#ifndef UPSAMPLE_NEAREST3D_SIMT
#define UPSAMPLE_NEAREST3D_SIMT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest3d_tiling_data.h"
#include "./upsample_nearest3d_simt_base.h"

namespace UpsampleNearest3d {
using namespace AscendC;

template <typename T1, typename T2, bool isExtra, uint64_t schId>
class Nearest3dSimt {
public:
    __aicore__ inline Nearest3dSimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UpsampleNearest3dRegBaseTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    const UpsampleNearest3dRegBaseTilingData *tilingData_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ inline void Nearest3dSimt<T1, T2, isExtra, schId>::Init(GM_ADDR x, GM_ADDR y,
    const UpsampleNearest3dRegBaseTilingData *__restrict tilingData)
{
    inputGm_.SetGlobalBuffer((__gm__ T1 *)x);
    outputGm_.SetGlobalBuffer((__gm__ T1 *)y);
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ inline void Nearest3dSimt<T1, T2, isExtra, schId>::Process()
{
    if (blockIdx_ >= static_cast<int32_t>(GetBlockNum())) {
        return;
    }
    T2 blkProcessNum = tilingData_->blkProcessNum;
    T2 blkStartOffset = blockIdx_ * blkProcessNum;
    if (blockIdx_ < tilingData_->tailBlockNum) {
        blkProcessNum = blkProcessNum + 1;
        blkStartOffset = blkStartOffset + blockIdx_;
    } else {
        blkStartOffset = blkStartOffset + tilingData_->tailBlockNum;
    }
    T2 mW = 0, shiftW = 0, mH = 0, shiftH = 0, mD = 0, shiftD = 0;
    T2 lenDstW = static_cast<T2>(tilingData_->outW);
    T2 lenDstH = static_cast<T2>(tilingData_->outH);
    T2 lenDstD = static_cast<T2>(tilingData_->outD);
    T2 lenN = static_cast<T2>(tilingData_->lenN);
    T2 lenC = static_cast<T2>(tilingData_->lenC);
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    if constexpr (schId != SCH_ID_1) {
        GetUintDivMagicAndShift(mD, shiftD, lenDstD);
    }
    T2 lenSrcW = static_cast<T2>(tilingData_->inW);
    T2 lenSrcH = static_cast<T2>(tilingData_->inH);
    T2 lenSrcD = static_cast<T2>(tilingData_->inD);
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        Simt::VF_CALL<calleeInt64<T1, T2, isExtra, schId>>(Simt::Dim3(THREAD_NUM_B64),
            (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()), blkStartOffset, blkProcessNum,
            lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW,
            tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW);
    } else {
        Simt::VF_CALL<calleeInt32<T1, T2, isExtra, schId>>(Simt::Dim3(THREAD_NUM_B32),
            (__gm__ T1 *)(inputGm_.GetPhyAddr()), (__gm__ T1 *)(outputGm_.GetPhyAddr()), blkStartOffset, blkProcessNum,
            lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW,
            tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW);
    }
}
} // namespace UpsampleNearest3d

#endif // UPSAMPLE_NEAREST3D_SIMT
