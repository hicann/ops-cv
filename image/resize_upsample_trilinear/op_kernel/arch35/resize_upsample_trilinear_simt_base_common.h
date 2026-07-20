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
 * \file resize_upsample_trilinear_simt_base_common.h
 * \brief ResizeUpsampleTrilinear SIMT shared constants, index/weight helpers and
 *        special-value handling primitives for arch35. Included by the other
 *        simt_base_*.h siblings; not meant to be included directly.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_COMMON_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_COMMON_H_

#include "./resize_upsample_trilinear_tiling_data.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

// 512 threads keeps the SIMT register footprint/occupancy balanced on A950.
// 1024 increases per-VF state pressure and regresses full 3D workloads.
constexpr int32_t THREAD_NUM_B32 = 512;
constexpr int32_t THREAD_NUM_B64 = 512;
// The 2x D-only specialization has a smaller register footprint than full 3D.
// Its separate 1024-thread VF retains memory-level parallelism without raising
// register pressure for other SIMT schedules.
constexpr int32_t THREAD_NUM_D_ONLY_B32 = 1024;
// The fixed full-3D path keeps H/W interpolation state and two cached D
// planes live across the 256-output D loop.  1024 threads leave too few
// registers per thread on A950 and can spill this state.  512 threads trade a
// longer per-thread HW loop for substantially better VF occupancy.
constexpr int32_t THREAD_NUM_NC_HW = 512;
constexpr int32_t THREAD_NUM_D_REUSE_NC_HW = 512;
constexpr float HALF_PIXEL = 0.5f;
constexpr int32_t INTERP_NEIGHBOR_SIZE = 2;

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeSourceIndex(float scale, T2 dstIndex,
                                                                                          int32_t alignCorners)
{
    if (alignCorners == 1) {
        return fmaf(static_cast<float>(dstIndex), scale, 0.0f);
    }
    float srcIndex = fmaf(static_cast<float>(dstIndex) + HALF_PIXEL, scale, -HALF_PIXEL);
    return srcIndex < 0.0f ? 0.0f : srcIndex;
}

template <bool AlignCorners, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeSourceIndexMode(float scale, T2 dstIndex)
{
    if constexpr (AlignCorners) {
        return fmaf(static_cast<float>(dstIndex), scale, 0.0f);
    }
    float srcIndex = fmaf(static_cast<float>(dstIndex) + HALF_PIXEL, scale, -HALF_PIXEL);
    return srcIndex < 0.0f ? 0.0f : srcIndex;
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeLinearIndexAndWeight(
    float realIndex, T2 limit, T2& index0, T2& index1, float& weight0, float& weight1)
{
    index0 = min(static_cast<T2>(realIndex), limit);
    index1 = min(static_cast<T2>(index0 + 1), limit);
    weight1 = min(max(realIndex - static_cast<float>(index0), 0.0f), 1.0f);
    weight0 = 1.0f - weight1;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool IsZeroWeightSpecialValue(float inputValue,
                                                                                               float weightD,
                                                                                               float weightH,
                                                                                               float weightW)
{
    return (weightD == 0.0f || weightH == 0.0f || weightW == 0.0f) && (isinf(inputValue) || isnan(inputValue));
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeOutIndex(T2 yGmIdx, T2 mW, T2 shiftW,
                                                                                      T2 lenDstW, T2 mH, T2 shiftH,
                                                                                      T2 lenDstH, T2 mD, T2 shiftD,
                                                                                      T2 lenDstD, T2& outW, T2& outH,
                                                                                      T2& outD, T2& nc)
{
    T2 tmpW = Simt::UintDiv(yGmIdx, mW, shiftW);
    outW = yGmIdx - tmpW * lenDstW;
    T2 tmpH = Simt::UintDiv(tmpW, mH, shiftH);
    outH = tmpW - tmpH * lenDstH;
    nc = Simt::UintDiv(tmpH, mD, shiftD);
    outD = tmpH - nc * lenDstD;
}

template <typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeInterpParams(
    float scaleD, T2 outD, T2 lenSrcD, float scaleH, T2 outH, T2 lenSrcH, float scaleW, T2 outW, T2 lenSrcW,
    int32_t alignCorners, T2& inD0, T2& inD1, T2& inH0, T2& inH1, T2& inW0, T2& inW1, float& weightD0, float& weightD1,
    float& weightH0, float& weightH1, float& weightW0, float& weightW1)
{
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleD, outD, alignCorners), lenSrcD - 1, inD0, inD1, weightD0,
                                weightD1);
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleH, outH, alignCorners), lenSrcH - 1, inH0, inH1, weightH0,
                                weightH1);
    ComputeLinearIndexAndWeight(ComputeSourceIndex(scaleW, outW, alignCorners), lenSrcW - 1, inW0, inW1, weightW0,
                                weightW1);
}

// 单轴计算+special值检查（axisMode: 0=D-only, 1=H-only, 2=W-only）
// 非活动维度权重传0，等价于：任何inf/nan输入→输出NAN（与参考实现一致）
template <typename T1, typename T2, int32_t AxisMode>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeSingleAxisFused(
    __gm__ T1* inputGm, __gm__ T1* outputGm, T2 yGmIdx, T2 addr0, T2 addr1, float w0, float w1)
{
    float v0 = static_cast<float>(inputGm[addr0]);
    float v1 = static_cast<float>(inputGm[addr1]);
    bool special;
    if constexpr (AxisMode == 0) {
        special = IsZeroWeightSpecialValue(v0, w0, 0.0f, 0.0f) || IsZeroWeightSpecialValue(v1, w1, 0.0f, 0.0f);
    } else if constexpr (AxisMode == 1) {
        special = IsZeroWeightSpecialValue(v0, 0.0f, w0, 0.0f) || IsZeroWeightSpecialValue(v1, 0.0f, w1, 0.0f);
    } else {
        special = IsZeroWeightSpecialValue(v0, 0.0f, 0.0f, w0) || IsZeroWeightSpecialValue(v1, 0.0f, 0.0f, w1);
    }
    float value = special ? ASCRT_NAN_F : fmaf(w0, v0, w1 * v1);
    outputGm[yGmIdx] = static_cast<T1>(value);
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void LoadInputValues(
    __gm__ T1* inputGm, T2 ncOffset, T2 inD0, T2 inD1, T2 inH0, T2 inH1, T2 inW0, T2 inW1, T2 lenSrcHw, T2 lenSrcW,
    float& value000, float& value001, float& value010, float& value011, float& value100, float& value101,
    float& value110, float& value111)
{
    T2 d0Offset = ncOffset + inD0 * lenSrcHw;
    T2 d1Offset = ncOffset + inD1 * lenSrcHw;
    T2 h00Offset = d0Offset + inH0 * lenSrcW;
    T2 h01Offset = d0Offset + inH1 * lenSrcW;
    T2 h10Offset = d1Offset + inH0 * lenSrcW;
    T2 h11Offset = d1Offset + inH1 * lenSrcW;
    value000 = static_cast<float>(inputGm[h00Offset + inW0]);
    value001 = static_cast<float>(inputGm[h00Offset + inW1]);
    value010 = static_cast<float>(inputGm[h01Offset + inW0]);
    value011 = static_cast<float>(inputGm[h01Offset + inW1]);
    value100 = static_cast<float>(inputGm[h10Offset + inW0]);
    value101 = static_cast<float>(inputGm[h10Offset + inW1]);
    value110 = static_cast<float>(inputGm[h11Offset + inW0]);
    value111 = static_cast<float>(inputGm[h11Offset + inW1]);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool CheckSpecialZeroWeight(
    float value000, float value001, float value010, float value011, float value100, float value101, float value110,
    float value111, float weightD0, float weightD1, float weightH0, float weightH1, float weightW0, float weightW1)
{
    return IsZeroWeightSpecialValue(value000, weightD0, weightH0, weightW0) ||
           IsZeroWeightSpecialValue(value001, weightD0, weightH0, weightW1) ||
           IsZeroWeightSpecialValue(value010, weightD0, weightH1, weightW0) ||
           IsZeroWeightSpecialValue(value011, weightD0, weightH1, weightW1) ||
           IsZeroWeightSpecialValue(value100, weightD1, weightH0, weightW0) ||
           IsZeroWeightSpecialValue(value101, weightD1, weightH0, weightW1) ||
           IsZeroWeightSpecialValue(value110, weightD1, weightH1, weightW0) ||
           IsZeroWeightSpecialValue(value111, weightD1, weightH1, weightW1);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeTrilinearValue(
    float weightD0, float weightD1, float weightH0, float weightH1, float weightW0, float weightW1, float value000,
    float value001, float value010, float value011, float value100, float value101, float value110, float value111)
{
    return weightD0 * (weightH0 * (weightW0 * value000 + weightW1 * value001) +
                       weightH1 * (weightW0 * value010 + weightW1 * value011)) +
           weightD1 * (weightH0 * (weightW0 * value100 + weightW1 * value101) +
                       weightH1 * (weightW0 * value110 + weightW1 * value111));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeFullFromValues(
    float wD0, float wD1, float wH0, float wH1, float wW0, float wW1, float v000, float v001, float v010, float v011,
    float v100, float v101, float v110, float v111)
{
    if (CheckSpecialZeroWeight(v000, v001, v010, v011, v100, v101, v110, v111, wD0, wD1, wH0, wH1, wW0, wW1)) {
        return ASCRT_NAN_F;
    }
    return ComputeTrilinearValue(wD0, wD1, wH0, wH1, wW0, wW1, v000, v001, v010, v011, v100, v101, v110, v111);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeBilinearPlane(float wH0, float wH1,
                                                                                            float wW0, float wW1,
                                                                                            float v00, float v01,
                                                                                            float v10, float v11)
{
    return wH0 * (wW0 * v00 + wW1 * v01) + wH1 * (wW0 * v10 + wW1 * v11);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool IsSpecialValue(float value)
{
    return isinf(value) || isnan(value);
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void LoadBilinearPlaneChecked(
    __gm__ T1* inputGm, T2 planeOffset, T2 h0Offset, T2 h1Offset, T2 inW0, T2 inW1, float wH0, float wH1, float wW0,
    float wW1, float& bilinear, uint32_t& specialFlags)
{
    float v00 = inputGm[planeOffset + h0Offset + inW0];
    float v01 = inputGm[planeOffset + h0Offset + inW1];
    float v10 = inputGm[planeOffset + h1Offset + inW0];
    float v11 = inputGm[planeOffset + h1Offset + inW1];
    bool s00 = IsSpecialValue(v00);
    bool s01 = IsSpecialValue(v01);
    bool s10 = IsSpecialValue(v10);
    bool s11 = IsSpecialValue(v11);
    bool hasSpecial = s00 || s01 || s10 || s11;
    bool hasZeroHwSpecial = ((wH0 == 0.0f || wW0 == 0.0f) && s00) || ((wH0 == 0.0f || wW1 == 0.0f) && s01) ||
                            ((wH1 == 0.0f || wW0 == 0.0f) && s10) || ((wH1 == 0.0f || wW1 == 0.0f) && s11);
    specialFlags = static_cast<uint32_t>(hasZeroHwSpecial) | (static_cast<uint32_t>(hasSpecial) << 1);
    bilinear = ComputeBilinearPlane(wH0, wH1, wW0, wW1, v00, v01, v10, v11);
}

} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_COMMON_H_
