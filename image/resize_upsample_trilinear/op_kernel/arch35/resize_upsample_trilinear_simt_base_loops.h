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
 * \file resize_upsample_trilinear_simt_base_loops.h
 * \brief Generic stride-based SIMT loops (AllOne / D-only / D-only 2x / H-only /
 *        W-only / Full) and shared coordinate-update helpers for arch35.
 *        Depends on resize_upsample_trilinear_simt_base_common.h.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_LOOPS_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_LOOPS_H_

#include "./resize_upsample_trilinear_simt_base_common.h"

namespace ResizeUpsampleTrilinear {

// All-scale-1路径：纯搬运（scale全为1.0时输入输出shape完全相同，直接地址搬运）
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopAllOne(__gm__ T1* inputGm,
                                                                                     __gm__ T1* outputGm,
                                                                                     T2 blkStartOffset,
                                                                                     T2 blkProcessNum)
{
    T2 stride = static_cast<T2>(blockDim.x);
    T2 yGmIdx = blkStartOffset + static_cast<T2>(threadIdx.x);
    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += stride) {
        outputGm[yGmIdx] = inputGm[yGmIdx];
        yGmIdx += stride;
    }
}

// 坐标增量更新辅助函数：carryHBase/remW由调用方预计算并传入（循环不变量外提）
template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CoordsIncrement(T2& outW, T2& outH, T2& outD,
                                                                                      T2& nc, T2 carryHBase, T2 remW,
                                                                                      T2 lenDstW, T2 lenDstH,
                                                                                      T2 lenDstD, T2 mH, T2 shiftH,
                                                                                      T2 mD, T2 shiftD)
{
    outW += remW;
    T2 addH = carryHBase;
    if (outW >= lenDstW) {
        outW -= lenDstW;
        addH++;
    }
    outH += addH;
    if (outH >= lenDstH) {
        T2 tmpD = Simt::UintDiv(outH, mH, shiftH);
        outH -= tmpD * lenDstH;
        outD += tmpD;
        if (outD >= lenDstD) {
            T2 tmpNc = Simt::UintDiv(outD, mD, shiftD);
            outD -= tmpNc * lenDstD;
            nc += tmpNc;
        }
    }
}

// D-only路径：只有D方向缩放（性能瓶颈路径）
// 关键优化：stride遍历+地址增量+插值参数缓存+CoordsIncrement+循环不变量外提
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopDOnly(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, T2 lenSrcHw, T2 lenSrcDhw,
    float scaleD, int32_t alignCorners)
{
    T2 stride = static_cast<T2>(blockDim.x);
    T2 yGmIdx = blkStartOffset + static_cast<T2>(threadIdx.x);
    T2 outW = 0, outH = 0, outD = 0, nc = 0;
    ComputeOutIndex(yGmIdx, mW, shiftW, lenDstW, mH, shiftH, lenDstH, mD, shiftD, lenDstD, outW, outH, outD, nc);

    // 循环不变量外提
    T2 carryHBase = stride / lenDstW;
    T2 remW = stride - carryHBase * lenDstW;

    T2 prevOutD = static_cast<T2>(-1);
    T2 prevNc = static_cast<T2>(-1);
    T2 i0 = 0, i1 = 0;
    float w0 = 0.0f, w1 = 0.0f;
    T2 addr0 = 0, addr1 = 0;
    T2 ncDeltaBase = (lenSrcD - lenDstD) * lenSrcHw;

    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += stride) {
        if (outD != prevOutD || nc != prevNc) {
            if (outD != prevOutD) {
                ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleD, outD, alignCorners), lenSrcD - 1, i0, i1, w0,
                                            w1);
                prevOutD = outD;
            }
            // i0/outD are unsigned; (i0 - outD) can be negative when upscaled,
            // so compute the delta in signed arithmetic to avoid wraparound.
            int64_t delta = static_cast<int64_t>(nc) * static_cast<int64_t>(ncDeltaBase) +
                            (static_cast<int64_t>(i0) - static_cast<int64_t>(outD)) * static_cast<int64_t>(lenSrcHw);
            addr0 = yGmIdx + static_cast<T2>(delta);
            addr1 = addr0 + static_cast<T2>((static_cast<int64_t>(i1) - static_cast<int64_t>(i0)) *
                                            static_cast<int64_t>(lenSrcHw));
            prevNc = nc;
        }
        ComputeSingleAxisFused<T1, T2, 0>(inputGm, outputGm, yGmIdx, addr0, addr1, w0, w1);

        yGmIdx += stride;
        addr0 += stride;
        addr1 += stride;
        CoordsIncrement<T2>(outW, outH, outD, nc, carryHBase, remW, lenDstW, lenDstH, lenDstD, mH, shiftH, mD, shiftD);
    }
}

// D-only 2x specialization used by the common scale_factor=(2,1,1) path.
// A thread owns one H/W coordinate and emits the two adjacent output planes,
// reusing both source values and avoiding generic output-coordinate division.
template <typename T1, typename T2, bool AlignCorners>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopDOnly2x(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 taskStart, T2 taskCount, T2 mSrcD, T2 shiftSrcD, T2 mSrcHw,
    T2 shiftSrcHw, T2 lenSrcD, T2 lenSrcHw, T2 lenDstHw, float scaleD)
{
    T2 stride = static_cast<T2>(blockDim.x);
    for (T2 localTask = static_cast<T2>(threadIdx.x); localTask < taskCount; localTask += stride) {
        T2 task = taskStart + localTask;
        T2 srcD = Simt::UintDiv(task, mSrcHw, shiftSrcHw);
        T2 hw = task - srcD * lenSrcHw;
        T2 nc = Simt::UintDiv(srcD, mSrcD, shiftSrcD);
        srcD -= nc * lenSrcD;
        T2 inputNcOffset = nc * lenSrcD * lenSrcHw;
        T2 outputNcOffset = nc * (lenSrcD << 1) * lenDstHw;

        T2 outD0 = srcD << 1;
        T2 outD1 = outD0 + 1;
        float src0 = ComputeSourceIndexMode<AlignCorners>(scaleD, outD0);
        float src1 = ComputeSourceIndexMode<AlignCorners>(scaleD, outD1);
        T2 in00 = 0, in01 = 0, in10 = 0, in11 = 0;
        float w00 = 0.0f, w01 = 0.0f, w10 = 0.0f, w11 = 0.0f;
        ComputeLinearIndexAndWeight(src0, lenSrcD - 1, in00, in01, w00, w01);
        ComputeLinearIndexAndWeight(src1, lenSrcD - 1, in10, in11, w10, w11);

        T2 outOffset0 = outputNcOffset + outD0 * lenDstHw + hw;
        T2 outOffset1 = outputNcOffset + outD1 * lenDstHw + hw;
        T2 inOffset00 = inputNcOffset + in00 * lenSrcHw + hw;
        T2 inOffset01 = inputNcOffset + in01 * lenSrcHw + hw;
        T2 inOffset10 = inputNcOffset + in10 * lenSrcHw + hw;
        T2 inOffset11 = inputNcOffset + in11 * lenSrcHw + hw;
        ComputeSingleAxisFused<T1, T2, 0>(inputGm, outputGm, outOffset0, inOffset00, inOffset01, w00, w01);
        ComputeSingleAxisFused<T1, T2, 0>(inputGm, outputGm, outOffset1, inOffset10, inOffset11, w10, w11);
    }
}

// H-only路径：只有H方向缩放
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopHOnly(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, T2 lenSrcHw, T2 lenSrcDhw,
    float scaleH, int32_t alignCorners)
{
    T2 stride = static_cast<T2>(blockDim.x);
    T2 yGmIdx = blkStartOffset + static_cast<T2>(threadIdx.x);
    T2 outW = 0, outH = 0, outD = 0, nc = 0;
    ComputeOutIndex(yGmIdx, mW, shiftW, lenDstW, mH, shiftH, lenDstH, mD, shiftD, lenDstD, outW, outH, outD, nc);

    // 循环不变量外提
    T2 carryHBase = stride / lenDstW;
    T2 remW = stride - carryHBase * lenDstW;

    T2 prevOutH = static_cast<T2>(-1);
    T2 prevOutD = static_cast<T2>(-1);
    T2 i0 = 0, i1 = 0;
    float w0 = 0.0f, w1 = 0.0f;
    T2 addr0 = 0, addr1 = 0;

    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += stride) {
        if (outH != prevOutH || outD != prevOutD) {
            if (outH != prevOutH) {
                ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleH, outH, alignCorners), lenSrcH - 1, i0, i1, w0,
                                            w1);
                prevOutH = outH;
            }
            // (i0 - outH) can be negative when upscaled; use signed arithmetic.
            int64_t delta = (static_cast<int64_t>(i0) - static_cast<int64_t>(outH)) * static_cast<int64_t>(lenSrcW);
            addr0 = yGmIdx + static_cast<T2>(delta);
            addr1 = addr0 + static_cast<T2>((static_cast<int64_t>(i1) - static_cast<int64_t>(i0)) *
                                            static_cast<int64_t>(lenSrcW));
            prevOutD = outD;
        }
        ComputeSingleAxisFused<T1, T2, 1>(inputGm, outputGm, yGmIdx, addr0, addr1, w0, w1);

        yGmIdx += stride;
        addr0 += stride;
        addr1 += stride;
        CoordsIncrement<T2>(outW, outH, outD, nc, carryHBase, remW, lenDstW, lenDstH, lenDstD, mH, shiftH, mD, shiftD);
    }
}

// W-only路径：只有W方向缩放
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopWOnly(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, T2 lenSrcHw, T2 lenSrcDhw, float scaleW,
    int32_t alignCorners)
{
    T2 stride = static_cast<T2>(blockDim.x);
    T2 yGmIdx = blkStartOffset + static_cast<T2>(threadIdx.x);
    T2 outW = 0, outH = 0, outD = 0, nc = 0;
    ComputeOutIndex(yGmIdx, mW, shiftW, lenDstW, mH, shiftH, lenDstH, mD, shiftD, lenDstD, outW, outH, outD, nc);

    // 循环不变量外提
    T2 carryHBase = stride / lenDstW;
    T2 remW = stride - carryHBase * lenDstW;

    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += stride) {
        T2 i0 = 0, i1 = 0;
        float w0 = 0.0f, w1 = 0.0f;
        ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleW, outW, alignCorners), lenSrcW - 1, i0, i1, w0, w1);
        // (i0 - outW) can be negative when upscaled; use signed arithmetic.
        int64_t delta0 = static_cast<int64_t>(i0) - static_cast<int64_t>(outW);
        T2 addr0 = yGmIdx + static_cast<T2>(delta0);
        T2 addr1 = addr0 + static_cast<T2>(static_cast<int64_t>(i1) - static_cast<int64_t>(i0));
        ComputeSingleAxisFused<T1, T2, 2>(inputGm, outputGm, yGmIdx, addr0, addr1, w0, w1);

        yGmIdx += stride;
        CoordsIncrement<T2>(outW, outH, outD, nc, carryHBase, remW, lenDstW, lenDstH, lenDstD, mH, shiftH, mD, shiftD);
    }
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline float LoadFullValue(__gm__ T1* inputGm, T2 h00Offset,
                                                                                     T2 h01Offset, T2 h10Offset,
                                                                                     T2 h11Offset, T2 inW0, T2 inW1,
                                                                                     float wD0, float wD1, float wH0,
                                                                                     float wH1, float wW0, float wW1)
{
    float v000 = static_cast<float>(inputGm[h00Offset + inW0]);
    float v001 = static_cast<float>(inputGm[h00Offset + inW1]);
    float v010 = static_cast<float>(inputGm[h01Offset + inW0]);
    float v011 = static_cast<float>(inputGm[h01Offset + inW1]);
    float v100 = static_cast<float>(inputGm[h10Offset + inW0]);
    float v101 = static_cast<float>(inputGm[h10Offset + inW1]);
    float v110 = static_cast<float>(inputGm[h11Offset + inW0]);
    float v111 = static_cast<float>(inputGm[h11Offset + inW1]);
    if (CheckSpecialZeroWeight(v000, v001, v010, v011, v100, v101, v110, v111, wD0, wD1, wH0, wH1, wW0, wW1)) {
        return ASCRT_NAN_F;
    }
    return ComputeTrilinearValue(wD0, wD1, wH0, wH1, wW0, wW1, v000, v001, v010, v011, v100, v101, v110, v111);
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateFullDhOffsets(
    T2 outD, T2 outH, T2 nc, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenSrcHw, T2 lenSrcDhw, float scaleD, float scaleH,
    int32_t alignCorners, T2& prevOutD, T2& prevOutH, T2& prevNc, T2& inD0, T2& inD1, T2& inH0, T2& inH1, float& wD0,
    float& wD1, float& wH0, float& wH1, T2& h00Offset, T2& h01Offset, T2& h10Offset, T2& h11Offset)
{
    if (outD != prevOutD) {
        ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleD, outD, alignCorners), lenSrcD - 1, inD0, inD1, wD0, wD1);
        prevOutD = outD;
    }
    if (outH != prevOutH) {
        ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleH, outH, alignCorners), lenSrcH - 1, inH0, inH1, wH0, wH1);
        prevOutH = outH;
    }
    T2 ncOffset = nc * lenSrcDhw;
    T2 d0Offset = ncOffset + inD0 * lenSrcHw;
    T2 d1Offset = ncOffset + inD1 * lenSrcHw;
    h00Offset = d0Offset + inH0 * lenSrcW;
    h01Offset = d0Offset + inH1 * lenSrcW;
    h10Offset = d1Offset + inH0 * lenSrcW;
    h11Offset = d1Offset + inH1 * lenSrcW;
    prevNc = nc;
}

// Full trilinear path.
template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopFull(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 blkStartOffset, T2 blkProcessNum, T2 mD, T2 shiftD, T2 mH, T2 shiftH,
    T2 mW, T2 shiftW, T2 lenSrcD, T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, T2 lenSrcHw, T2 lenSrcDhw,
    float scaleD, float scaleH, float scaleW, int32_t alignCorners)
{
    T2 stride = static_cast<T2>(blockDim.x);
    T2 yGmIdx = blkStartOffset + static_cast<T2>(threadIdx.x);
    T2 outW = 0, outH = 0, outD = 0, nc = 0;
    ComputeOutIndex(yGmIdx, mW, shiftW, lenDstW, mH, shiftH, lenDstH, mD, shiftD, lenDstD, outW, outH, outD, nc);

    // 循环不变量外提
    T2 carryHBase = stride / lenDstW;
    T2 remW = stride - carryHBase * lenDstW;

    T2 prevOutD = static_cast<T2>(-1);
    T2 prevOutH = static_cast<T2>(-1);
    T2 prevNc = static_cast<T2>(-1);
    T2 inD0 = 0, inD1 = 0, inH0 = 0, inH1 = 0;
    float wD0 = 0.0f, wD1 = 0.0f, wH0 = 0.0f, wH1 = 0.0f;
    T2 h00Offset = 0, h01Offset = 0, h10Offset = 0, h11Offset = 0;

    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += stride) {
        bool dhChanged = (outD != prevOutD || outH != prevOutH);
        bool ncChanged = (nc != prevNc);
        if (dhChanged || ncChanged) {
            UpdateFullDhOffsets(outD, outH, nc, lenSrcD, lenSrcH, lenSrcW, lenSrcHw, lenSrcDhw, scaleD, scaleH,
                                alignCorners, prevOutD, prevOutH, prevNc, inD0, inD1, inH0, inH1, wD0, wD1, wH0, wH1,
                                h00Offset, h01Offset, h10Offset, h11Offset);
        }
        T2 inW0 = 0, inW1 = 0;
        float wW0 = 0.0f, wW1 = 0.0f;
        ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleW, outW, alignCorners), lenSrcW - 1, inW0, inW1, wW0, wW1);
        outputGm[yGmIdx] = static_cast<T1>(LoadFullValue(inputGm, h00Offset, h01Offset, h10Offset, h11Offset, inW0,
                                                         inW1, wD0, wD1, wH0, wH1, wW0, wW1));

        yGmIdx += stride;
        CoordsIncrement<T2>(outW, outH, outD, nc, carryHBase, remW, lenDstW, lenDstH, lenDstD, mH, shiftH, mD, shiftD);
    }
}

} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_LOOPS_H_
