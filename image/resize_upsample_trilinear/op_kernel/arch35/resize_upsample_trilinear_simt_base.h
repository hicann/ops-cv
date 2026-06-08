/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_simt_base.h
 * \brief SIMT compute base functions for ResizeUpsampleTrilinear
 */
#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

const int32_t THREAD_NUM_B32 = 512;
const int32_t THREAD_NUM_B64 = 512;

static __simt_callee__ __aicore__ __attribute__((always_inline)) inline float CalcSourceCoord(
    int outIdx, float scale, bool alignCorners)
{
    if (alignCorners) {
        return static_cast<float>(outIdx) * scale;
    }
    return fmaxf((static_cast<float>(outIdx) + 0.5f) * scale - 0.5f, 0.0f);
}

static __simt_callee__ __aicore__ __attribute__((always_inline)) inline int ClampIndex(
    int idx, int maxVal)
{
    return (idx < 0) ? 0 : (idx > maxVal) ? maxVal : idx;
}

template <typename T2>
struct DepthInterpCoeffs {
    int x0;
    int x1;
    float lambda0;
    float lambda1;
    T2 addr_d0;
    T2 addr_d1;
};

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline DepthInterpCoeffs<T2> CalcDepthInterp(
    float src_d, int maxD, T2 input_base, T2 stride_d_input)
{
    DepthInterpCoeffs<T2> c;
    c.x0 = ClampIndex(static_cast<int>(floorf(src_d)), maxD);
    c.x1 = (c.x0 + 1 > maxD) ? maxD : c.x0 + 1;
    c.lambda0 = static_cast<float>(c.x1) - src_d;
    c.lambda1 = 1.0f - c.lambda0;
    c.addr_d0 = input_base + static_cast<T2>(c.x0) * stride_d_input;
    c.addr_d1 = input_base + static_cast<T2>(c.x1) * stride_d_input;
    return c;
}

template <typename T2>
struct HeightInterpCoeffs {
    int y0;
    int y1;
    float mu0;
    float mu1;
    T2 addr_d0_h0;
    T2 addr_d0_h1;
    T2 addr_d1_h0;
    T2 addr_d1_h1;
};

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline HeightInterpCoeffs<T2> CalcHeightInterp(
    float src_h, int maxH, T2 addr_d0, T2 addr_d1, T2 stride_h_input)
{
    HeightInterpCoeffs<T2> c;
    c.y0 = ClampIndex(static_cast<int>(floorf(src_h)), maxH);
    c.y1 = (c.y0 + 1 > maxH) ? maxH : c.y0 + 1;
    c.mu0 = static_cast<float>(c.y1) - src_h;
    c.mu1 = 1.0f - c.mu0;
    c.addr_d0_h0 = addr_d0 + static_cast<T2>(c.y0) * stride_h_input;
    c.addr_d0_h1 = addr_d0 + static_cast<T2>(c.y1) * stride_h_input;
    c.addr_d1_h0 = addr_d1 + static_cast<T2>(c.y0) * stride_h_input;
    c.addr_d1_h1 = addr_d1 + static_cast<T2>(c.y1) * stride_h_input;
    return c;
}

struct CombinedCoeffs {
    float lambda0_mu0;
    float lambda0_mu1;
    float lambda1_mu0;
    float lambda1_mu1;
};

static __simt_callee__ __aicore__ __attribute__((always_inline)) inline CombinedCoeffs CalcCombinedCoeffs(
    float lambda0, float lambda1, float mu0, float mu1)
{
    CombinedCoeffs c;
    c.lambda0_mu0 = lambda0 * mu0;
    c.lambda0_mu1 = lambda0 * mu1;
    c.lambda1_mu0 = lambda1 * mu0;
    c.lambda1_mu1 = lambda1 * mu1;
    return c;
}

template <typename T2>
struct SimtComputeParams {
    T2 input_d;
    T2 input_h;
    T2 input_w;
    T2 output_d;
    T2 output_h;
    T2 output_w;
    float scale_d;
    float scale_h;
    float scale_w;
    bool alignCorners;
    float half_scale_w;
    int maxD;
    int maxH;
    int maxW;
    T2 stride_bc_input;
    T2 stride_d_input;
    T2 stride_h_input;
    T2 stride_bc_output;
    T2 stride_d_output;
    T2 stride_h_output;
    T2 plane_size;
    T2 row_size;
};

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline SimtComputeParams<T2> InitSimtParams(
    const ResizeUpsampleTrilinearArch35TilingData* __restrict tilingData)
{
    SimtComputeParams<T2> p;
    p.input_d = static_cast<T2>(tilingData->input_d);
    p.input_h = static_cast<T2>(tilingData->input_h);
    p.input_w = static_cast<T2>(tilingData->input_w);
    p.output_d = static_cast<T2>(tilingData->output_d);
    p.output_h = static_cast<T2>(tilingData->output_h);
    p.output_w = static_cast<T2>(tilingData->output_w);
    p.scale_d = tilingData->scale_d;
    p.scale_h = tilingData->scale_h;
    p.scale_w = tilingData->scale_w;
    p.alignCorners = (tilingData->align_corners != 0);
    p.half_scale_w = 0.5f * p.scale_w;
    p.maxD = static_cast<int>(p.input_d - 1);
    p.maxH = static_cast<int>(p.input_h - 1);
    p.maxW = static_cast<int>(p.input_w - 1);
    p.stride_bc_input = p.input_d * p.input_h * p.input_w;
    p.stride_d_input = p.input_h * p.input_w;
    p.stride_h_input = p.input_w;
    p.stride_bc_output = p.output_d * p.output_h * p.output_w;
    p.stride_d_output = p.output_h * p.output_w;
    p.stride_h_output = p.output_w;
    p.plane_size = p.output_d * p.output_h * p.output_w;
    p.row_size = p.output_h * p.output_w;
    return p;
}

template <typename T2>
struct SimtThreadState {
    T2 bc;
    int od;
    int oh;
    int ow;
    T2 flat_idx;
    T2 input_base;
    T2 output_base;
    float src_d;
    float src_h;
    DepthInterpCoeffs<T2> depthCoeffs;
    HeightInterpCoeffs<T2> heightCoeffs;
    CombinedCoeffs combinedCoeffs;
};

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline SimtThreadState<T2> InitThreadState(
    T2 blkStartOffset, T2 threadOffset, const SimtComputeParams<T2>& p,
    const ResizeUpsampleTrilinearArch35TilingData* __restrict tilingData)
{
    SimtThreadState<T2> s;
    s.flat_idx = blkStartOffset + threadOffset;
    s.bc = s.flat_idx / p.plane_size;
    T2 spatial_idx = s.flat_idx - s.bc * p.plane_size;
    s.od = static_cast<int>(spatial_idx / p.row_size);
    T2 hw_idx = spatial_idx - static_cast<T2>(s.od) * p.row_size;
    s.oh = static_cast<int>(hw_idx / p.output_w);
    s.ow = static_cast<int>(hw_idx - static_cast<T2>(s.oh) * p.output_w);
    s.input_base = s.bc * p.stride_bc_input;
    s.output_base = s.bc * p.stride_bc_output + static_cast<T2>(s.od) * p.stride_d_output +
                    static_cast<T2>(s.oh) * p.stride_h_output;
    s.src_d = CalcSourceCoord(s.od, p.scale_d, p.alignCorners);
    s.src_h = CalcSourceCoord(s.oh, p.scale_h, p.alignCorners);
    s.depthCoeffs = CalcDepthInterp<T2>(s.src_d, p.maxD, s.input_base, p.stride_d_input);
    s.heightCoeffs = CalcHeightInterp<T2>(s.src_h, p.maxH, s.depthCoeffs.addr_d0, s.depthCoeffs.addr_d1, p.stride_h_input);
    s.combinedCoeffs = CalcCombinedCoeffs(s.depthCoeffs.lambda0, s.depthCoeffs.lambda1,
                                           s.heightCoeffs.mu0, s.heightCoeffs.mu1);
    return s;
}

template <typename T1, typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline float ComputeTrilinearValue(
    __gm__ T1* input, const HeightInterpCoeffs<T2>& hc, int z0, int z1,
    float nu0, float nu1, const CombinedCoeffs& cc)
{
    T1 raw_v000 = asc_ldcg(&input[hc.addr_d0_h0 + z0]);
    T1 raw_v001 = asc_ldcg(&input[hc.addr_d0_h0 + z1]);
    T1 raw_v010 = asc_ldcg(&input[hc.addr_d0_h1 + z0]);
    T1 raw_v011 = asc_ldcg(&input[hc.addr_d0_h1 + z1]);
    T1 raw_v100 = asc_ldcg(&input[hc.addr_d1_h0 + z0]);
    T1 raw_v101 = asc_ldcg(&input[hc.addr_d1_h0 + z1]);
    T1 raw_v110 = asc_ldcg(&input[hc.addr_d1_h1 + z0]);
    T1 raw_v111 = asc_ldcg(&input[hc.addr_d1_h1 + z1]);

    float v000, v001, v010, v011, v100, v101, v110, v111;
    if constexpr (std::is_same<T1, half>::value || std::is_same<T1, bfloat16_t>::value) {
        v000 = static_cast<float>(raw_v000); v001 = static_cast<float>(raw_v001);
        v010 = static_cast<float>(raw_v010); v011 = static_cast<float>(raw_v011);
        v100 = static_cast<float>(raw_v100); v101 = static_cast<float>(raw_v101);
        v110 = static_cast<float>(raw_v110); v111 = static_cast<float>(raw_v111);
    } else {
        v000 = raw_v000; v001 = raw_v001;
        v010 = raw_v010; v011 = raw_v011;
        v100 = raw_v100; v101 = raw_v101;
        v110 = raw_v110; v111 = raw_v111;
    }

    return v000 * cc.lambda0_mu0 * nu0 + v001 * cc.lambda0_mu0 * nu1
         + v010 * cc.lambda0_mu1 * nu0 + v011 * cc.lambda0_mu1 * nu1
         + v100 * cc.lambda1_mu0 * nu0 + v101 * cc.lambda1_mu0 * nu1
         + v110 * cc.lambda1_mu1 * nu0 + v111 * cc.lambda1_mu1 * nu1;
}

template <typename T1, typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline void StoreOutputValue(
    __gm__ T1* output, T2 output_addr, float result)
{
    if constexpr (std::is_same<T1, half>::value || std::is_same<T1, bfloat16_t>::value) {
        T1 output_val = static_cast<T1>(result);
        asc_stcg(&output[output_addr], output_val);
    } else {
        asc_stcg(&output[output_addr], result);
    }
}

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateCoordsOnNewBatch(
    SimtThreadState<T2>& s, const SimtComputeParams<T2>& p)
{
    s.input_base = s.bc * p.stride_bc_input;
    s.output_base = s.bc * p.stride_bc_output;
    s.src_d = CalcSourceCoord(0, p.scale_d, p.alignCorners);
    s.src_h = CalcSourceCoord(0, p.scale_h, p.alignCorners);
    s.depthCoeffs = CalcDepthInterp<T2>(s.src_d, p.maxD, s.input_base, p.stride_d_input);
    s.heightCoeffs = CalcHeightInterp<T2>(s.src_h, p.maxH, s.depthCoeffs.addr_d0, s.depthCoeffs.addr_d1, p.stride_h_input);
    s.combinedCoeffs = CalcCombinedCoeffs(s.depthCoeffs.lambda0, s.depthCoeffs.lambda1,
                                           s.heightCoeffs.mu0, s.heightCoeffs.mu1);
}

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateCoordsOnNewDepth(
    SimtThreadState<T2>& s, const SimtComputeParams<T2>& p)
{
    s.src_d = CalcSourceCoord(s.od, p.scale_d, p.alignCorners);
    s.depthCoeffs = CalcDepthInterp<T2>(s.src_d, p.maxD, s.input_base, p.stride_d_input);
    s.heightCoeffs = CalcHeightInterp<T2>(s.src_h, p.maxH, s.depthCoeffs.addr_d0, s.depthCoeffs.addr_d1, p.stride_h_input);
    s.combinedCoeffs = CalcCombinedCoeffs(s.depthCoeffs.lambda0, s.depthCoeffs.lambda1,
                                           s.heightCoeffs.mu0, s.heightCoeffs.mu1);
    s.output_base = s.bc * p.stride_bc_output + static_cast<T2>(s.od) * p.stride_d_output;
}

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline void UpdateCoordsOnNewHeight(
    SimtThreadState<T2>& s, const SimtComputeParams<T2>& p)
{
    s.src_h = CalcSourceCoord(s.oh, p.scale_h, p.alignCorners);
    s.heightCoeffs = CalcHeightInterp<T2>(s.src_h, p.maxH, s.depthCoeffs.addr_d0, s.depthCoeffs.addr_d1, p.stride_h_input);
    s.combinedCoeffs = CalcCombinedCoeffs(s.depthCoeffs.lambda0, s.depthCoeffs.lambda1,
                                           s.heightCoeffs.mu0, s.heightCoeffs.mu1);
    s.output_base = s.bc * p.stride_bc_output + static_cast<T2>(s.od) * p.stride_d_output +
                    static_cast<T2>(s.oh) * p.stride_h_output;
}

template <typename T2>
static __simt_callee__ __aicore__ __attribute__((always_inline)) inline void AdvanceOutputCoords(
    SimtThreadState<T2>& s, const SimtComputeParams<T2>& p)
{
    s.ow++;
    s.flat_idx++;
    if (s.ow >= p.output_w) {
        s.ow = 0;
        s.oh++;
        if (s.oh >= p.output_h) {
            s.oh = 0;
            s.od++;
            if (s.od >= p.output_d) {
                s.od = 0;
                s.bc++;
                UpdateCoordsOnNewBatch(s, p);
            } else {
                UpdateCoordsOnNewDepth(s, p);
            }
        } else {
            UpdateCoordsOnNewHeight(s, p);
        }
    }
}

template <typename T1, typename T2>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCompute(
    __gm__ T1* output, __gm__ T1* input, T2 blkStartOffset, T2 blkProcessNum,
    const ResizeUpsampleTrilinearArch35TilingData* __restrict tilingData)
{
    SimtComputeParams<T2> p = InitSimtParams<T2>(tilingData);
    uint32_t elements_per_thread = tilingData->elements_per_thread;

    T2 tid = static_cast<T2>(threadIdx.x);
    T2 threadOffset = tid * static_cast<T2>(elements_per_thread);

    if (threadOffset >= blkProcessNum) {
        return;
    }

    T2 elementsToProcess = static_cast<T2>(elements_per_thread);
    T2 remaining = blkProcessNum - threadOffset;
    if (remaining < elementsToProcess) {
        elementsToProcess = remaining;
    }

    SimtThreadState<T2> s = InitThreadState<T2>(blkStartOffset, threadOffset, p, tilingData);
    if (s.flat_idx >= static_cast<T2>(tilingData->total_elements)) {
        return;
    }

    for (T2 e = 0; e < elementsToProcess; e++) {
        if (s.flat_idx >= static_cast<T2>(tilingData->total_elements)) {
            break;
        }

        float src_w;
        if (p.alignCorners) {
            src_w = static_cast<float>(s.ow) * p.scale_w;
        } else {
            src_w = fmaxf(static_cast<float>(s.ow) * p.scale_w + p.half_scale_w - 0.5f, 0.0f);
        }

        int z0 = ClampIndex(static_cast<int>(floorf(src_w)), p.maxW);
        int z1 = (z0 + 1 > p.maxW) ? p.maxW : z0 + 1;
        float nu0 = static_cast<float>(z1) - src_w;
        float nu1 = 1.0f - nu0;

        float result = ComputeTrilinearValue<T1, T2>(input, s.heightCoeffs, z0, z1, nu0, nu1, s.combinedCoeffs);
        T2 output_addr = s.output_base + static_cast<T2>(s.ow);
        StoreOutputValue<T1, T2>(output, output_addr, result);

        AdvanceOutputCoords(s, p);
    }
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B32) __aicore__ void calleeInt32(
    __gm__ T1* output, __gm__ T1* input, T2 blkStartOffset, T2 blkProcessNum,
    const ResizeUpsampleTrilinearArch35TilingData* __restrict tilingData)
{
    SimtCompute<T1, T2>(output, input, blkStartOffset, blkProcessNum, tilingData);
}

template <typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_B64) __aicore__ void calleeInt64(
    __gm__ T1* output, __gm__ T1* input, T2 blkStartOffset, T2 blkProcessNum,
    const ResizeUpsampleTrilinearArch35TilingData* __restrict tilingData)
{
    SimtCompute<T1, T2>(output, input, blkStartOffset, blkProcessNum, tilingData);
}
} // namespace ResizeUpsampleTrilinear
#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_BASE_H