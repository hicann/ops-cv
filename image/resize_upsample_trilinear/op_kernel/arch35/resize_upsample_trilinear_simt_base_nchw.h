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
 * \file resize_upsample_trilinear_simt_base_nchw.h
 * \brief NC/HW specialized SIMT paths (D-reuse walk and fixed 256^3 hotspot)
 *        for arch35. Depends on resize_upsample_trilinear_simt_base_common.h.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_NCHW_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_NCHW_H_

#include "./resize_upsample_trilinear_simt_base_common.h"

namespace ResizeUpsampleTrilinear {

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateDReusePlanes(
    __gm__ T1* inputGm, T2 inputNcOffset, T2 lenSrcHw, T2 h0Offset, T2 h1Offset, T2 inW0, T2 inW1, float wH0, float wH1,
    float wW0, float wW1, T2 inD0, T2 inD1, T2& cachedD0, T2& cachedD1, float& bilinear0, float& bilinear1,
    uint32_t& plane0SpecialFlags, uint32_t& plane1SpecialFlags)
{
    if (inD0 == cachedD1) {
        bilinear0 = bilinear1;
        plane0SpecialFlags = plane1SpecialFlags;
    } else {
        LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + inD0 * lenSrcHw, h0Offset, h1Offset, inW0, inW1, wH0,
                                         wH1, wW0, wW1, bilinear0, plane0SpecialFlags);
    }
    if (inD1 == inD0) {
        bilinear1 = bilinear0;
        plane1SpecialFlags = plane0SpecialFlags;
    } else {
        LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + inD1 * lenSrcHw, h0Offset, h1Offset, inW0, inW1, wH0,
                                         wH1, wW0, wW1, bilinear1, plane1SpecialFlags);
    }
    cachedD0 = inD0;
    cachedD1 = inD1;
}

// Generic large-output path. A thread owns one (N*C, H, W) coordinate and
// walks D in order, so H/W parameters are reused and adjacent D planes stay
// cached in registers. Neighboring threads still issue contiguous output
// stores for each D plane.
template <typename T1, typename T2, bool AlignCorners>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeDReuseTask(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 taskIdx, T2 mW, T2 shiftW, T2 mH, T2 shiftH, T2 lenSrcD, T2 lenSrcH,
    T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD, float scaleH, float scaleW)
{
    T2 lenSrcHw = lenSrcH * lenSrcW;
    T2 lenSrcDhw = lenSrcD * lenSrcHw;
    T2 lenDstHw = lenDstH * lenDstW;
    T2 lenDstDhw = lenDstD * lenDstHw;
    T2 tmpW = Simt::UintDiv(taskIdx, mW, shiftW);
    T2 outW = taskIdx - tmpW * lenDstW;
    T2 nc = Simt::UintDiv(tmpW, mH, shiftH);
    T2 outH = tmpW - nc * lenDstH;
    T2 inH0 = 0, inH1 = 0, inW0 = 0, inW1 = 0;
    float wH0 = 0.0f, wH1 = 0.0f, wW0 = 0.0f, wW1 = 0.0f;
    ComputeLinearIndexAndWeight(ComputeSourceIndexMode<AlignCorners>(scaleH, outH), lenSrcH - 1, inH0, inH1, wH0, wH1);
    ComputeLinearIndexAndWeight(ComputeSourceIndexMode<AlignCorners>(scaleW, outW), lenSrcW - 1, inW0, inW1, wW0, wW1);
    T2 h0Offset = inH0 * lenSrcW;
    T2 h1Offset = inH1 * lenSrcW;
    T2 inputNcOffset = nc * lenSrcDhw;
    T2 outputNcOffset = nc * lenDstDhw;
    T2 cachedD0 = static_cast<T2>(-1), cachedD1 = static_cast<T2>(-1);
    float bilinear0 = 0.0f, bilinear1 = 0.0f;
    uint32_t plane0SpecialFlags = 0, plane1SpecialFlags = 0;
    for (T2 outD = 0; outD < lenDstD; ++outD) {
        T2 inD0 = 0, inD1 = 0;
        float wD0 = 0.0f, wD1 = 0.0f;
        ComputeLinearIndexAndWeight(ComputeSourceIndexMode<AlignCorners>(scaleD, outD), lenSrcD - 1, inD0, inD1, wD0,
                                    wD1);
        if (inD0 != cachedD0 || inD1 != cachedD1) {
            UpdateDReusePlanes<T1, T2>(inputGm, inputNcOffset, lenSrcHw, h0Offset, h1Offset, inW0, inW1, wH0, wH1, wW0,
                                       wW1, inD0, inD1, cachedD0, cachedD1, bilinear0, bilinear1, plane0SpecialFlags,
                                       plane1SpecialFlags);
        }
        bool special = ((plane0SpecialFlags | plane1SpecialFlags) & 1U) != 0U ||
                       (wD0 == 0.0f && (plane0SpecialFlags & 2U) != 0U) ||
                       (wD1 == 0.0f && (plane1SpecialFlags & 2U) != 0U);
        float value = special ? ASCRT_NAN_F : wD0 * bilinear0 + wD1 * bilinear1;
        outputGm[outputNcOffset + outD * lenDstHw + outH * lenDstW + outW] = static_cast<T1>(value);
    }
}

template <typename T1, typename T2, bool AlignCorners>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopDReuseNcHw(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 taskStart, T2 taskCount, T2 mW, T2 shiftW, T2 mH, T2 shiftH, T2 lenSrcD,
    T2 lenSrcH, T2 lenSrcW, T2 lenDstD, T2 lenDstH, T2 lenDstW, float scaleD, float scaleH, float scaleW)
{
    T2 stride = static_cast<T2>(blockDim.x);
    for (T2 localTask = static_cast<T2>(threadIdx.x); localTask < taskCount; localTask += stride) {
        ComputeDReuseTask<T1, T2, AlignCorners>(inputGm, outputGm, taskStart + localTask, mW, shiftW, mH, shiftH,
                                                lenSrcD, lenSrcH, lenSrcW, lenDstD, lenDstH, lenDstW, scaleD, scaleH,
                                                scaleW);
    }
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeHalfScaleHwParams(T2 outIndex, T2& in0,
                                                                                               T2& in1, float& w0,
                                                                                               float& w1)
{
    if (outIndex == 0) {
        in0 = 0;
        in1 = 1;
        w1 = 0.0f;
    } else {
        in0 = (outIndex - 1) >> 1;
        in1 = min(in0 + 1, static_cast<T2>(127));
        w1 = (outIndex & 1) == 0 ? 0.75f : 0.25f;
    }
    w0 = 1.0f - w1;
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateFullNcHwAlignPlanes(
    __gm__ T1* inputGm, T2 inputNcOffset, T2 h0Offset, T2 h1Offset, T2 inW0, T2 inW1, float wH0, float wH1, float wW0,
    float wW1, T2 inD0, T2 inD1, T2& cachedD0, T2& cachedD1, float& bilinear0, float& bilinear1,
    uint32_t& plane0SpecialFlags, uint32_t& plane1SpecialFlags)
{
    constexpr int32_t SRC_D_SHIFT = 14;
    if (inD0 == cachedD1) {
        bilinear0 = bilinear1;
        plane0SpecialFlags = plane1SpecialFlags;
    } else {
        LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + (inD0 << SRC_D_SHIFT), h0Offset, h1Offset, inW0, inW1,
                                         wH0, wH1, wW0, wW1, bilinear0, plane0SpecialFlags);
    }
    if (inD1 == inD0) {
        bilinear1 = bilinear0;
        plane1SpecialFlags = plane0SpecialFlags;
    } else {
        LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + (inD1 << SRC_D_SHIFT), h0Offset, h1Offset, inW0, inW1,
                                         wH0, wH1, wW0, wW1, bilinear1, plane1SpecialFlags);
    }
    cachedD0 = inD0;
    cachedD1 = inD1;
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void FullNcHwAlignDepth(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 inputNcOffset, T2 outputNcOffset, T2 hwGmIdx, T2 h0Offset, T2 h1Offset,
    T2 inW0, T2 inW1, float wH0, float wH1, float wW0, float wW1, float scaleD, int32_t alignCorners)
{
    constexpr T2 SRC_D_LIMIT = static_cast<T2>(7);
    constexpr int32_t SRC_D_SHIFT = 14;
    constexpr T2 DST_HW_MASK = static_cast<T2>(255);
    constexpr int32_t DST_D_SHIFT = 16;
    T2 cachedD0 = static_cast<T2>(-1), cachedD1 = static_cast<T2>(-1);
    float bilinear0 = 0.0f, bilinear1 = 0.0f;
    uint32_t plane0SpecialFlags = 0, plane1SpecialFlags = 0;
    for (T2 outD = 0; outD <= DST_HW_MASK; ++outD) {
        T2 inD0 = 0, inD1 = 0;
        float wD0 = 0.0f, wD1 = 0.0f;
        ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleD, outD, alignCorners), SRC_D_LIMIT, inD0, inD1, wD0, wD1);
        if (inD0 != cachedD0 || inD1 != cachedD1) {
            UpdateFullNcHwAlignPlanes<T1, T2>(inputGm, inputNcOffset, h0Offset, h1Offset, inW0, inW1, wH0, wH1, wW0,
                                              wW1, inD0, inD1, cachedD0, cachedD1, bilinear0, bilinear1,
                                              plane0SpecialFlags, plane1SpecialFlags);
        }
        bool special = ((plane0SpecialFlags | plane1SpecialFlags) & 1U) != 0U ||
                       (wD0 == 0.0f && (plane0SpecialFlags & 2U) != 0U) ||
                       (wD1 == 0.0f && (plane1SpecialFlags & 2U) != 0U);
        float value = special ? ASCRT_NAN_F : wD0 * bilinear0 + wD1 * bilinear1;
        outputGm[outputNcOffset + (outD << DST_D_SHIFT) + hwGmIdx] = static_cast<T1>(value);
    }
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void FullNcHwHalfDepth(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 inputNcOffset, T2 outputNcOffset, T2 hwGmIdx, T2 h0Offset, T2 h1Offset,
    T2 inW0, T2 inW1, float wH0, float wH1, float wW0, float wW1, float scaleD, int32_t alignCorners)
{
    constexpr T2 SRC_D_LIMIT = static_cast<T2>(7);
    constexpr int32_t SRC_D_SHIFT = 14;
    constexpr int32_t DST_D_SHIFT = 16;
    float bilinear0 = 0.0f, bilinear1 = 0.0f;
    uint32_t plane0SpecialFlags = 0, plane1SpecialFlags = 0;
    LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset, h0Offset, h1Offset, inW0, inW1, wH0, wH1, wW0, wW1,
                                     bilinear0, plane0SpecialFlags);
    LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + (static_cast<T2>(1) << SRC_D_SHIFT), h0Offset, h1Offset,
                                     inW0, inW1, wH0, wH1, wW0, wW1, bilinear1, plane1SpecialFlags);
    bool firstSpecial = ((plane0SpecialFlags | plane1SpecialFlags) & 1U) != 0U || (plane1SpecialFlags & 2U) != 0U;
    float firstW1 = ComputeSourceIndex(scaleD, static_cast<T2>(0), alignCorners);
    float firstW0 = 1.0f - firstW1;
    for (T2 outD = 0; outD < static_cast<T2>(16); ++outD) {
        float value = firstSpecial ? ASCRT_NAN_F : firstW0 * bilinear0 + firstW1 * bilinear1;
        outputGm[outputNcOffset + (outD << DST_D_SHIFT) + hwGmIdx] = static_cast<T1>(value);
    }
    for (T2 pair = 0; pair < SRC_D_LIMIT; ++pair) {
        if (pair != 0) {
            bilinear0 = bilinear1;
            plane0SpecialFlags = plane1SpecialFlags;
            LoadBilinearPlaneChecked<T1, T2>(inputGm, inputNcOffset + ((pair + 1) << SRC_D_SHIFT), h0Offset, h1Offset,
                                             inW0, inW1, wH0, wH1, wW0, wW1, bilinear1, plane1SpecialFlags);
        }
        bool pairSpecial = ((plane0SpecialFlags | plane1SpecialFlags) & 1U) != 0U;
        T2 outDBase = static_cast<T2>(16) + (pair << 5);
        for (T2 offset = 0; offset < static_cast<T2>(32); ++offset) {
            float wD1 = static_cast<float>((offset << 1) + 1) * 0.015625f;
            float wD0 = 1.0f - wD1;
            float value = pairSpecial ? ASCRT_NAN_F : wD0 * bilinear0 + wD1 * bilinear1;
            outputGm[outputNcOffset + ((outDBase + offset) << DST_D_SHIFT) + hwGmIdx] = static_cast<T1>(value);
        }
    }
    bilinear0 = bilinear1;
    bool lastSpecial = (plane1SpecialFlags & 1U) != 0U;
    for (T2 offset = 0; offset < static_cast<T2>(16); ++offset) {
        float wD1 = static_cast<float>((offset << 1) + 1) * 0.015625f;
        float wD0 = 1.0f - wD1;
        float value = lastSpecial ? ASCRT_NAN_F : wD0 * bilinear0 + wD1 * bilinear0;
        outputGm[outputNcOffset + ((static_cast<T2>(240) + offset) << DST_D_SHIFT) + hwGmIdx] = static_cast<T1>(value);
    }
}

// NC/HW path for the fixed 256^3 acc.json hotspot.
template <typename T1, typename T2, bool AlignCorners>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtLoopFullNcHw(__gm__ T1* inputGm,
                                                                                       __gm__ T1* outputGm, T2 nc,
                                                                                       T2 hwProcessNum, float scaleD,
                                                                                       float scaleH, float scaleW,
                                                                                       int32_t alignCorners)
{
    constexpr T2 SRC_HW_MASK = static_cast<T2>(127);
    constexpr int32_t SRC_H_SHIFT = 7;
    constexpr int32_t SRC_NC_SHIFT = 17;
    constexpr T2 DST_HW_MASK = static_cast<T2>(255);
    constexpr int32_t DST_H_SHIFT = 8;
    constexpr int32_t DST_NC_SHIFT = 24;
    T2 stride = static_cast<T2>(blockDim.x);
    T2 inputNcOffset = nc << SRC_NC_SHIFT;
    T2 outputNcOffset = nc << DST_NC_SHIFT;
    T2 hwGmIdx = static_cast<T2>(threadIdx.x);
    for (T2 hwLocalIdx = static_cast<T2>(threadIdx.x); hwLocalIdx < hwProcessNum; hwLocalIdx += stride) {
        T2 outW = hwGmIdx & DST_HW_MASK;
        T2 outH = hwGmIdx >> DST_H_SHIFT;
        T2 inH0 = 0, inH1 = 0, inW0 = 0, inW1 = 0;
        float wH0 = 0.0f, wH1 = 0.0f, wW0 = 0.0f, wW1 = 0.0f;
        if constexpr (AlignCorners) {
            ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleH, outH, alignCorners), SRC_HW_MASK, inH0, inH1, wH0,
                                        wH1);
            ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleW, outW, alignCorners), SRC_HW_MASK, inW0, inW1, wW0,
                                        wW1);
        } else {
            ComputeHalfScaleHwParams(outH, inH0, inH1, wH0, wH1);
            ComputeHalfScaleHwParams(outW, inW0, inW1, wW0, wW1);
        }
        if constexpr (AlignCorners) {
            FullNcHwAlignDepth<T1, T2>(inputGm, outputGm, inputNcOffset, outputNcOffset, hwGmIdx, inH0 << SRC_H_SHIFT,
                                       inH1 << SRC_H_SHIFT, inW0, inW1, wH0, wH1, wW0, wW1, scaleD, alignCorners);
        } else {
            FullNcHwHalfDepth<T1, T2>(inputGm, outputGm, inputNcOffset, outputNcOffset, hwGmIdx, inH0 << SRC_H_SHIFT,
                                      inH1 << SRC_H_SHIFT, inW0, inW1, wH0, wH1, wW0, wW1, scaleD, alignCorners);
        }
        hwGmIdx += stride;
    }
}

} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_NCHW_H_
