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
 * \file resize_upsample_trilinear_simt_base.h
 * \brief ResizeUpsampleTrilinear SIMT compute base for arch35.
 *        Aggregator header: pulls in the common helpers, the generic
 *        stride-based loops and the NC/HW specialized paths, then defines the
 *        unified SimtCompute dispatcher and the __simt_vf__ entry functions.
 *        Split out of the former monolithic header for readability; the
 *        generated kernel symbols and inlining behavior are unchanged.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_

#include "./resize_upsample_trilinear_simt_base_common.h"
#include "./resize_upsample_trilinear_simt_base_loops.h"
#include "./resize_upsample_trilinear_simt_base_nchw.h"

namespace ResizeUpsampleTrilinear {

// 统一入口：根据scale参数分发到专用路径，保留所有优化
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD,
    float scaleH, float scaleW, int32_t alignCorners)
{
    bool dEq = (scaleD == 1.0f);
    bool hEq = (scaleH == 1.0f);
    bool wEq = (scaleW == 1.0f);
    T2 lenSrcHw = lenSrcH * lenSrcW;
    T2 lenSrcDhw = lenSrcD * lenSrcHw;

    if (dEq && hEq && wEq) {
        SimtLoopAllOne<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum);
    } else if (hEq && wEq) {
        SimtLoopDOnly<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW,
                              lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, lenSrcHw, lenSrcDhw, scaleD,
                              alignCorners);
    } else if (dEq && wEq) {
        SimtLoopHOnly<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW,
                              lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, lenSrcHw, lenSrcDhw, scaleH, alignCorners);
    } else if (dEq && hEq) {
        SimtLoopWOnly<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW,
                              lenSrcW, lenDstD, lenDstH, lenDstW, lenSrcHw, lenSrcDhw, scaleW, alignCorners);
    } else {
        SimtLoopFull<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW,
                             lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, lenSrcHw, lenSrcDhw, scaleD, scaleH,
                             scaleW, alignCorners);
    }
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__
    void calleeInt32(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH,
                     T2 lenDstW, float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
                        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}

template <typename T1, typename T2, bool AlignCorners>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_D_ONLY_B32) __aicore__
    void calleeInt32DOnly2x(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 taskStart, T2 taskCount, T2 mSrcD, T2 shiftSrcD,
                            T2 mSrcHw, T2 shiftSrcHw, T2 lenSrcD, T2 lenSrcHw, T2 lenDstHw, float scaleD)
{
    SimtLoopDOnly2x<T1, T2, AlignCorners>(inputGm, outputGm, taskStart, taskCount, mSrcD, shiftSrcD, mSrcHw, shiftSrcHw,
                                          lenSrcD, lenSrcHw, lenDstHw, scaleD);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__
    void calleeInt64(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD,
                     T2 mH, T2 shiftH, T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH,
                     T2 lenDstW, float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    SimtCompute<T1, T2>(inputGm, outputGm, blkStartOffset, blkProcessNum, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD,
                        lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW, alignCorners);
}

template <typename T1, typename T2, bool AlignCorners>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_NC_HW) __aicore__
    void calleeInt32NcHw(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 nc, T2 hwProcessNum, float scaleD, float scaleH,
                         float scaleW, int32_t alignCorners)
{
    SimtLoopFullNcHw<T1, T2, AlignCorners>(inputGm, outputGm, nc, hwProcessNum, scaleD, scaleH, scaleW, alignCorners);
}

template <typename T1, typename T2, bool AlignCorners>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_D_REUSE_NC_HW) __aicore__
    void calleeDReuseNcHw(__gm__ T1* inputGm, __gm__ T1* outputGm, T2 taskStart, T2 taskCount, T2 mW, T2 shiftW, T2 mH,
                          T2 shiftH, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW,
                          float scaleD, float scaleH, float scaleW)
{
    SimtLoopDReuseNcHw<T1, T2, AlignCorners>(inputGm, outputGm, taskStart, taskCount, mW, shiftW, mH, shiftH, lenSrcD,
                                             lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH, scaleW);
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H_
