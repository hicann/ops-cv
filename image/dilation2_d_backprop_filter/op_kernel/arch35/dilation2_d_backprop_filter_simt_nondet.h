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
 * \file dilation2_d_backprop_filter_simt_nondet.h
 * \brief SIMT kernel implementation for dilation2_d_backprop_filter (non-deterministic)
 *
 * Two-phase non-deterministic execution (aligned with TF GPU):
 *   Phase 1 - ZeroOut: grid-stride zero-fill yGm (output tensor)
 *   SyncAll - cross-core barrier (implicitly waits VF completion; volatile bypasses DCache)
 *   Phase 2 - Compute: Grid-Stride argmax search + asc_atomic_add to yGm
 *
 * Performance optimization:
 *   - IS_NCHW (bool template param): compile-time data format dispatch via if constexpr
 *   - IDX_T (typename template param): 32/64-bit index type selection
 *   - Phase 1 V_S sync + DCache flush removed: SyncAll implicitly waits VF;
 *     volatile qualifier on yGm forces direct GM writes (no DCache buffering)
 *   - Trailing V_S sync removed: kernel framework implicitly waits for VF completion
 *
 * Performance optimization:
 *   - IS_NCHW (bool template param): compile-time data format dispatch via if constexpr
 *   - IDX_T (typename template param): 32/64-bit index type selection
 */

#ifndef DILATION2D_BACKPROP_FILTER_SIMT_NONDET_H_
#define DILATION2D_BACKPROP_FILTER_SIMT_NONDET_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"
#include "dilation2_d_backprop_filter_tiling_data.h"
#include "dilation2_d_backprop_filter_tiling_key.h"

namespace NsDilation2DBackpropFilterNonDet {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_callee__ __aicore__ inline T GetLowestVal()
{
    return -ASCRT_MAX_NORMAL_F;
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ZeroOutSimt(int64_t filterSize, __gm__ volatile T* yGm)
{
    const uint64_t totalThreads = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    const uint64_t end = static_cast<uint64_t>(filterSize);
    for (uint64_t idx =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x);
         idx < end; idx += totalThreads) {
        yGm[idx] = static_cast<T>(0);
    }
}

template <typename T, bool IS_NCHW, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ComputeSimt(
    int64_t totalElements, int64_t inputH, int64_t inputW, int64_t depth, int64_t filterH, int64_t filterW,
    int64_t outH, int64_t outW, int64_t strideH, int64_t strideW, int64_t rateH, int64_t rateW, int64_t padTop,
    int64_t padLeft, int64_t padInputH, int64_t padInputW, __ubuf__ IDX_T* uintdivUb, __gm__ T* xGm, __gm__ T* filterGm,
    __gm__ T* outBackpropGm, __gm__ T* yGm)
{
    const IDX_T magic0 = uintdivUb[0];
    const IDX_T shift0 = uintdivUb[1];
    const IDX_T magic1 = uintdivUb[2];
    const IDX_T shift1 = uintdivUb[3];
    const IDX_T magic2 = uintdivUb[4];
    const IDX_T shift2 = uintdivUb[5];

    const int64_t xDepthStride = IS_NCHW ? (inputH * inputW) : 1;
    const int64_t xRowStride = IS_NCHW ? inputW : (inputW * depth);
    const int64_t xWStride = IS_NCHW ? 1 : depth;
    const int64_t inputBatchStride = inputH * inputW * depth;
    const int64_t filterDepthStride = IS_NCHW ? (filterH * filterW) : 1;
    const int64_t filterRowStride = IS_NCHW ? filterW : (filterW * depth);
    const int64_t filterWStride = IS_NCHW ? 1 : depth;

    IDX_T div0, div1, div2;
    if constexpr (IS_NCHW) {
        div0 = static_cast<IDX_T>(outW);
        div1 = static_cast<IDX_T>(outH);
        div2 = static_cast<IDX_T>(depth);
    } else {
        div0 = static_cast<IDX_T>(depth);
        div1 = static_cast<IDX_T>(outW);
        div2 = static_cast<IDX_T>(outH);
    }

    const T lowestVal = GetLowestVal<T>();
    const IDX_T gridStride = static_cast<IDX_T>(blockDim.x) * static_cast<IDX_T>(gridDim.x);

    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
         idx < static_cast<IDX_T>(totalElements); idx += gridStride) {
        IDX_T rem = idx;
        IDX_T q = Simt::UintDiv<IDX_T>(rem, magic0, shift0);
        IDX_T c0 = rem - q * div0;
        rem = q;
        q = Simt::UintDiv<IDX_T>(rem, magic1, shift1);
        IDX_T c1 = rem - q * div1;
        rem = q;
        q = Simt::UintDiv<IDX_T>(rem, magic2, shift2);
        IDX_T c2 = rem - q * div2;
        IDX_T b = q;

        IDX_T d, hOut, wOut;
        if constexpr (IS_NCHW) {
            wOut = c0;
            hOut = c1;
            d = c2;
        } else {
            d = c0;
            wOut = c1;
            hOut = c2;
        }

        int64_t hBeg = static_cast<int64_t>(hOut) * strideH - padTop;
        int64_t wBeg = static_cast<int64_t>(wOut) * strideW - padLeft;
        int64_t xBase = static_cast<int64_t>(b) * inputBatchStride + static_cast<int64_t>(d) * xDepthStride;
        int64_t filterBase = static_cast<int64_t>(d) * filterDepthStride;

        T curVal = lowestVal;
        int64_t hMax = 0;
        int64_t wMax = 0;
        for (int64_t h = 0; h < filterH; ++h) {
            int64_t hIn = hBeg + h * rateH;
            if (hIn >= 0 && hIn < padInputH) {
                int64_t xRowOffset = xBase + hIn * xRowStride;
                int64_t filterRowOffset = filterBase + h * filterRowStride;
                for (int64_t w = 0; w < filterW; ++w) {
                    int64_t wIn = wBeg + w * rateW;
                    if (wIn >= 0 && wIn < padInputW) {
                        T xVal;
                        if (hIn < inputH && wIn < inputW) {
                            xVal = xGm[xRowOffset + wIn * xWStride];
                        } else {
                            xVal = static_cast<T>(0);
                        }
                        T val = xVal + filterGm[filterRowOffset + w * filterWStride];
                        if (val > curVal) {
                            curVal = val;
                            hMax = h;
                            wMax = w;
                        }
                    }
                }
            }
        }

        int64_t yOffset = hMax * filterRowStride + wMax * filterWStride + static_cast<int64_t>(d) * filterDepthStride;
        T grad = outBackpropGm[idx];
        asc_atomic_add(yGm + yOffset, grad);
    }
}

template <typename T, bool IS_NCHW, typename IDX_T>
__aicore__ inline void LaunchComputeVf(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm,
                                       __gm__ T* filterGm, __gm__ T* outBackpropGm, __gm__ T* yGm)
{
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<IDX_T> ub = ubAlloc.Alloc<IDX_T>(8);

    IDX_T magic = 0;
    IDX_T shift = 0;
    if constexpr (IS_NCHW) {
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->outW));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->outH));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->depth));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    } else {
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->depth));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->outW));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);
        GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(tilingData->outH));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    }
    DataSyncBarrier<MemDsbT::UB>();
    __ubuf__ IDX_T* uintdivPtr = (__ubuf__ IDX_T*)ub.GetPhyAddr();

    asc_vf_call<ComputeSimt<T, IS_NCHW, IDX_T>>(
        dim3(THREAD_NUM), tilingData->totalElements, tilingData->inputH, tilingData->inputW, tilingData->depth,
        tilingData->filterH, tilingData->filterW, tilingData->outH, tilingData->outW, tilingData->strideH,
        tilingData->strideW, tilingData->rateH, tilingData->rateW, tilingData->padTop, tilingData->padLeft,
        tilingData->padInputH, tilingData->padInputW, uintdivPtr, xGm, filterGm, outBackpropGm, yGm);
}

template <typename T, bool IS_NCHW, typename IDX_T>
__aicore__ inline void Process(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm, __gm__ T* filterGm,
                               __gm__ T* outBackpropGm, __gm__ T* yGm)
{
    if (tilingData->totalElements <= 0 || tilingData->filterSize <= 0) {
        asc_vf_call<ZeroOutSimt<T>>(dim3(THREAD_NUM), tilingData->filterSize, yGm);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return;
    }

    asc_vf_call<ZeroOutSimt<T>>(dim3(THREAD_NUM), tilingData->filterSize, yGm);

    SyncAll();

    LaunchComputeVf<T, IS_NCHW, IDX_T>(tilingData, xGm, filterGm, outBackpropGm, yGm);
}

} // namespace NsDilation2DBackpropFilterNonDet
#endif // DILATION2D_BACKPROP_FILTER_SIMT_NONDET_H_
