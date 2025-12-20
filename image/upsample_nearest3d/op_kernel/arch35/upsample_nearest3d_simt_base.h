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
 * \file upsample_nearest3d_simt_base.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_SIMT_BASE_H
#define UPSAMPLE_NEAREST3D_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./upsample_nearest3d_tiling_data.h"

namespace UpsampleNearest3d {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 2048;
const int32_t THREAD_NUM_B64 = 1024;
const uint64_t SCH_ID_3 = 3;
const uint64_t SCH_ID_2 = 2;
const uint64_t SCH_ID_1 = 1;

template <typename T2, bool isExtra>
__aicore__ __attribute__((always_inline)) inline void ComputeOri(T2 idx, T2 limtData, float scale, T2 &origD)
{
    if constexpr (isExtra) {
        origD = Simt::Floor(((static_cast<float>(idx) + 0.5f) * scale));
    } else {
        origD = Simt::Floor((static_cast<float>(idx) * scale));
    }
    origD = Simt::Min(origD, limtData);
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__aicore__ __attribute__((always_inline)) inline void SimtCompute(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD, T2 mH, T2 shiftH, T2 mW, T2 shiftW,
    T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD, float scaleH, float scaleW)
{
    for (T2 idx = static_cast<T2>(Simt::GetThreadIdx()); idx < blkProcessNum;
        idx += static_cast<T2>(Simt::GetThreadNum<0>())) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 W = 0, H = 0, D = 0, C = 0, N = 0;
        T2 tmpRes = Simt::UintDiv(yGmIdx, mW, shiftW);
        W = yGmIdx - tmpRes * lenDstW;
        if constexpr (schId == SCH_ID_1) {
            D = Simt::UintDiv(tmpRes, mH, shiftH);
            H = tmpRes - D * lenDstH;
        }
        if constexpr (schId == SCH_ID_2) {
            T2 tmpRes1 = Simt::UintDiv(tmpRes, mH, shiftH);
            H = tmpRes - tmpRes1 * lenDstH;
            C = Simt::UintDiv(tmpRes1, mD, shiftD);
            D = tmpRes1 - C * lenDstD;
        }
        if constexpr (schId == SCH_ID_3) {
            T2 tmpRes1 = Simt::UintDiv(tmpRes, mH, shiftH);
            H = tmpRes - tmpRes1 * lenDstH;
            N = Simt::UintDiv(tmpRes1, mD, shiftD);
            D = tmpRes1 - N * lenDstD;
        }
        T2 origD = 0, origH = 0, origW = 0;
        ComputeOri<T2, isExtra>(D, lenSrcD - 1, scaleD, origD);
        ComputeOri<T2, isExtra>(H, lenSrcH - 1, scaleH, origH);
        ComputeOri<T2, isExtra>(W, lenSrcW - 1, scaleW, origW);
        T2 lenSrcHw = lenSrcH * lenSrcW;
        if constexpr (schId == SCH_ID_3) {
            T2 srcOffset = N * lenSrcD * lenSrcHw + origD * lenSrcHw + origH * lenSrcW + origW;
            outputGm[yGmIdx] = inputGm[srcOffset];
        }
        if constexpr (schId == SCH_ID_1) {
            for (T2 nc = 0; nc < lenN; nc++) {
                T2 outOffset = nc * lenDstD * lenDstH * lenDstW + yGmIdx;
                T2 srcOffset = nc * lenSrcD * lenSrcHw + origD * lenSrcHw + origH * lenSrcW + origW;
                outputGm[outOffset] = inputGm[srcOffset];
            }
        }
        if constexpr (schId == SCH_ID_2) {
            for (T2 n = 0; n < lenN; n++) {
                T2 ncIdx = n * lenC;
                T2 outOffset = ncIdx * lenDstD * lenDstH * lenDstW + yGmIdx;
                T2 lenSrcDhw = lenSrcD * lenSrcHw;
                T2 srcOffset = ncIdx * lenSrcDhw + C * lenSrcDhw + origD * lenSrcHw + origH * lenSrcW + origW;
                outputGm[outOffset] = inputGm[srcOffset];
            }
        }
    }
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32)__aicore__ void calleeInt32(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD, T2 mH, T2 shiftH, T2 mW, T2 shiftW,
    T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD, float scaleH, float scaleW)
{
    SimtCompute<T1, T2, isExtra, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH,
        shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW);
}

template <typename T1, typename T2, bool isExtra, uint64_t schId>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64)__aicore__ void calleeInt64(__gm__ T1 *inputGm, __gm__ T1 *outputGm,
    T2 blkStartOffset, T2 blkProcessNum, T2 lenN, T2 lenC, T2 mD, T2 shiftD, T2 mH, T2 shiftH, T2 mW, T2 shiftW,
    T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD, float scaleH, float scaleW)
{
    SimtCompute<T1, T2, isExtra, schId>(inputGm, outputGm, blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH,
        shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW);
}
} // namespace UpsampleNearest3d
#endif // UPSAMPLE_NEAREST3D_SIMT_BASE_H