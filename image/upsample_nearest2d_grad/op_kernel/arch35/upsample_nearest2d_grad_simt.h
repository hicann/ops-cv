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
 * \file upsample_nearest2d_grad_simt.h
 * \brief SIMT kernel implementation for upsample_nearest2d_grad operator
 *
 * Performance optimizations applied:
 *   R002: UintDiv fast division for coordinate decomposition (3 divisions → 2 UintDiv + multiply-subtract)
 *   R003: 32/64-bit index template (INDEX_SIZE_T) for grid-stride loop
 *   R006: Thread count tuned by index width (1024 for uint32_t, 512 for uint64_t)
 */

#ifndef UPSAMPLE_NEAREST2D_GRAD_SIMT_H_
#define UPSAMPLE_NEAREST2D_GRAD_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "upsample_nearest2d_grad_tiling_data.h"
#include "upsample_nearest2d_grad_tiling_key.h"

namespace NsUpsampleNearest2dGrad {

using namespace AscendC;

// R006: Thread count tuned by index width
// uint32_t path: 1024 threads (lower register pressure, higher occupancy)
// uint64_t path: 512 threads (higher register pressure from 64-bit ops)
template <typename INDEX_SIZE_T>
static constexpr uint32_t THREAD_NUM_OPT = (sizeof(INDEX_SIZE_T) == 4) ? 1024 : 512;

// === Helper functions: type-safe read/write ===

template <typename T>
__simt_callee__ __aicore__ inline float ReadAsFloat(__gm__ T* ptr, int64_t idx);

template <>
__simt_callee__ __aicore__ inline float ReadAsFloat<half>(__gm__ half* ptr, int64_t idx)
{
    return __half2float(ptr[idx]);
}

template <>
__simt_callee__ __aicore__ inline float ReadAsFloat<bfloat16_t>(__gm__ bfloat16_t* ptr, int64_t idx)
{
    return __bfloat162float(ptr[idx]);
}

template <>
__simt_callee__ __aicore__ inline float ReadAsFloat<float>(__gm__ float* ptr, int64_t idx)
{
    return ptr[idx];
}

template <typename T>
__simt_callee__ __aicore__ inline void WriteFromFloat(__gm__ T* ptr, int64_t idx, float val);

template <>
__simt_callee__ __aicore__ inline void WriteFromFloat<half>(__gm__ half* ptr, int64_t idx, float val)
{
    ptr[idx] = __float2half(val);
}

template <>
__simt_callee__ __aicore__ inline void WriteFromFloat<bfloat16_t>(__gm__ bfloat16_t* ptr, int64_t idx, float val)
{
    ptr[idx] = __float2bfloat16(val);
}

template <>
__simt_callee__ __aicore__ inline void WriteFromFloat<float>(__gm__ float* ptr, int64_t idx, float val)
{
    ptr[idx] = val;
}

// === Inner accumulation loop (split to keep function body <= 50 lines) ===

template <typename T>
__simt_callee__ __aicore__ inline float AccumulateGrad(__gm__ T* gradOutput, int32_t b, int64_t srcBase,
                                                       int64_t srcNCStride, int32_t srcYStart, int32_t srcYEnd,
                                                       int32_t srcXStart, int32_t srcXEnd, int32_t dimWout)
{
    float grad = 0.0f;
    for (int32_t y = srcYStart; y < srcYEnd; y++) {
        for (int32_t x = srcXStart; x < srcXEnd; x++) {
            int64_t srcIdx = (int64_t)b * srcNCStride + srcBase + (int64_t)y * dimWout + (int64_t)x;
            grad += ReadAsFloat<T>(gradOutput, srcIdx);
        }
    }
    return grad;
}

// === SIMT VF Kernel (R002: UintDiv, R003: INDEX_SIZE_T template, R006: thread tuning) ===

template <typename T, typename INDEX_SIZE_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_OPT<INDEX_SIZE_T>) inline void OpUpsampleNearest2dGradSimtKernel(
    INDEX_SIZE_T totalElements, int32_t dimN, int32_t dimC, int32_t dimHin, int32_t dimWin, int32_t dimHout,
    int32_t dimWout, float scaleH, float scaleW, INDEX_SIZE_T magic0, INDEX_SIZE_T shift0, INDEX_SIZE_T magic1,
    INDEX_SIZE_T shift1, __gm__ T* gradOutput, __gm__ T* gradInput)
{
    const INDEX_SIZE_T dstCStride = (INDEX_SIZE_T)dimHin * (INDEX_SIZE_T)dimWin;
    const int32_t srcCStride = dimHout * dimWout;
    const int64_t dstNCStride = (int64_t)dimC * (int64_t)dstCStride;
    const int64_t srcNCStride = (int64_t)dimC * (int64_t)srcCStride;

    for (INDEX_SIZE_T index = static_cast<INDEX_SIZE_T>(blockIdx.x * blockDim.x + threadIdx.x); index < totalElements;
         index += static_cast<INDEX_SIZE_T>(blockDim.x * gridDim.x)) {
        // R002: UintDiv replaces 3 hardware divisions with 2 fast-div + multiply-subtract
        INDEX_SIZE_T cIdxFull = Simt::UintDiv<INDEX_SIZE_T>(index, magic0, shift0);
        int32_t cIdx = static_cast<int32_t>(cIdxFull % static_cast<INDEX_SIZE_T>(dimC));
        INDEX_SIZE_T rem0 = index - cIdxFull * dstCStride;
        INDEX_SIZE_T hInIdx = Simt::UintDiv<INDEX_SIZE_T>(rem0, magic1, shift1);
        int32_t hIn = static_cast<int32_t>(hInIdx);
        int32_t wIn = static_cast<int32_t>(rem0 - hInIdx * (INDEX_SIZE_T)dimWin);

        int32_t srcYStart = static_cast<int32_t>(min(
            static_cast<long long int>(ceilf(static_cast<float>(hIn) * scaleH)), static_cast<long long int>(dimHout)));
        int32_t srcYEnd = static_cast<int32_t>(
            min(static_cast<long long int>(ceilf(static_cast<float>(hIn + 1) * scaleH)),
                static_cast<long long int>(dimHout)));
        int32_t srcXStart = static_cast<int32_t>(min(
            static_cast<long long int>(ceilf(static_cast<float>(wIn) * scaleW)), static_cast<long long int>(dimWout)));
        int32_t srcXEnd = static_cast<int32_t>(
            min(static_cast<long long int>(ceilf(static_cast<float>(wIn + 1) * scaleW)),
                static_cast<long long int>(dimWout)));

        int64_t dstIdx = static_cast<int64_t>(index);
        int64_t srcBase = (int64_t)cIdx * srcCStride;

        for (int32_t b = 0; b < dimN; b++) {
            float grad = AccumulateGrad<T>(gradOutput, b, srcBase, srcNCStride, srcYStart, srcYEnd, srcXStart, srcXEnd,
                                           dimWout);
            WriteFromFloat<T>(gradInput, dstIdx, grad);
            dstIdx += dstNCStride;
        }
    }
}

// === Process entry (R002: precompute magic/shift, R003: dispatch by index width) ===

template <typename T>
__aicore__ inline void Process(GM_ADDR gradOutput, GM_ADDR gradInput, const UpsampleNearest2dGradTilingData* tilingData)
{
    __gm__ T* gradOutputGm = (__gm__ T*)gradOutput;
    __gm__ T* gradInputGm = (__gm__ T*)gradInput;

    const int64_t totalElements = tilingData->totalElements;
    const bool use64Bit = (totalElements > static_cast<int64_t>(INT32_MAX));

    if (use64Bit) {
        using IDX_T = uint64_t;
        IDX_T magic0 = 0, shift0 = 0, magic1 = 0, shift1 = 0;
        IDX_T dstCStride = (IDX_T)tilingData->dimHin * (IDX_T)tilingData->dimWin;
        GetUintDivMagicAndShift<IDX_T>(magic0, shift0, dstCStride);
        GetUintDivMagicAndShift<IDX_T>(magic1, shift1, (IDX_T)tilingData->dimWin);
        asc_vf_call<OpUpsampleNearest2dGradSimtKernel<T, uint64_t>>(
            dim3(THREAD_NUM_OPT<uint64_t>), static_cast<IDX_T>(totalElements), tilingData->dimN, tilingData->dimC,
            tilingData->dimHin, tilingData->dimWin, tilingData->dimHout, tilingData->dimWout, tilingData->scaleH,
            tilingData->scaleW, magic0, shift0, magic1, shift1, gradOutputGm, gradInputGm);
    } else {
        using IDX_T = uint32_t;
        IDX_T magic0 = 0, shift0 = 0, magic1 = 0, shift1 = 0;
        IDX_T dstCStride = (IDX_T)tilingData->dimHin * (IDX_T)tilingData->dimWin;
        GetUintDivMagicAndShift<IDX_T>(magic0, shift0, dstCStride);
        GetUintDivMagicAndShift<IDX_T>(magic1, shift1, (IDX_T)tilingData->dimWin);
        asc_vf_call<OpUpsampleNearest2dGradSimtKernel<T, uint32_t>>(
            dim3(THREAD_NUM_OPT<uint32_t>), static_cast<IDX_T>(totalElements), tilingData->dimN, tilingData->dimC,
            tilingData->dimHin, tilingData->dimWin, tilingData->dimHout, tilingData->dimWout, tilingData->scaleH,
            tilingData->scaleW, magic0, shift0, magic1, shift1, gradOutputGm, gradInputGm);
    }
}

} // namespace NsUpsampleNearest2dGrad

#endif // UPSAMPLE_NEAREST2D_GRAD_SIMT_H_
