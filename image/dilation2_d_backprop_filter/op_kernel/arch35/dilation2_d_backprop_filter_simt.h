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
 * \file dilation2_d_backprop_filter_simt.h
 * \brief SIMT kernel implementation for dilation2_d_backprop_filter operator
 *
 * Three-phase deterministic execution:
 *   Phase 1  - ZeroOut: grid-stride zero-fill per-thread workspace
 *   Phase 2  - Compute: Grid-Stride argmax search + sequential += to per-thread buffer
 *   Phase 3a - ReduceThreads: tree reduction (1024 buffers → 1, O(10×filterSize))
 *   Phase 3b - ReduceCores: core 0 sequentially accumulates per-core buffers to yGm
 *
 * v2.4 SyncAll + tree reduction optimization:
 *   - SyncAll reduced from 8 to 2 (only cross-core dependencies)
 *   - yGm zero-out removed (Phase 3b writes all elements with `=`)
 *   - perCoreBuf zero-out removed (Phase 3a writes all elements with `=`)
 *   - Phase 2→3a same-core dependency (DCache clean suffices, no SyncAll)
 *   - Merged pre-Phase3b SyncAll pair into single SyncAll
 *   - ReduceThreadsSimt: tree reduction replaces sequential scan
 *     O(filterSize × 1024) → O(filterSize × log2(1024)) = O(filterSize × 10)
 *
 * v2.2 deterministic accumulation (per-thread buffer, no atomic_add):
 *   - Each thread (coreId, threadId) exclusively owns its filter-sized buffer
 *   - Multiple out_backprop positions mapping to same filter position handled by
 *     same thread via sequential += accumulation (fully deterministic)
 *   - Phase 3a tree reduction: fixed merge order, fully deterministic
 *
 * For each out_backprop position (b, h_out, w_out, d):
 *   1. Decompose linear index via UintDiv (divisors: depth, outW, outH)
 *   2. Scan filter window to find argmax of (input + filter)
 *   3. Sequential += out_backprop[idx] to perThreadBuf[hMax, wMax, d]
 */

#ifndef DILATION2D_BACKPROP_FILTER_SIMT_H_
#define DILATION2D_BACKPROP_FILTER_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"
#include "dilation2_d_backprop_filter_tiling_data.h"
#include "dilation2_d_backprop_filter_tiling_key.h"

namespace NsDilation2DBackpropFilter {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024; // MDE §2.2: 1024 threads (constexpr)

// ============================================================================
// GetLowestVal: returns lowest finite value for type T
// v2.0: only float supported, returns -FLT_MAX directly
// Equivalent to np.finfo(float).min (MDE §5.1, golden line 80)
// ============================================================================
template <typename T>
__simt_callee__ __aicore__ inline T GetLowestVal()
{
    return -ASCRT_MAX_NORMAL_F; // -FLT_MAX ≈ -3.4e38, from math_constants.h
}

// ============================================================================
// Phase 1: Zero-out filter_backprop (y) via Grid-Stride
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ZeroOutSimt(int64_t filterSize, __gm__ T* dstGm)
{
    const uint64_t totalThreads = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    const uint64_t end = static_cast<uint64_t>(filterSize);

    for (uint64_t idx =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x);
         idx < end; idx += totalThreads) {
        dstGm[idx] = static_cast<T>(0);
    }
}

// ============================================================================
// Phase 2: Compute argmax + sequential accumulation to per-thread buffer
//
// v2.2: fully deterministic — per-thread buffer eliminates ALL atomic_add
//   - Each thread (coreId, threadId) writes to its own filter-sized buffer:
//     wsBuf[coreId * THREAD_NUM * perCoreBufElems + threadId * perCoreBufElems + yOffset]
//   - No atomic_add needed: each thread exclusively owns its buffer region
//   - Multiple out_backprop positions mapping to the same filter position
//     are handled by the SAME thread via sequential += accumulation
//   - Grid-Stride ensures each thread processes its own slice of totalElements
//
// Phase 3a: Reduce threads → per-core buffer (each core reduces its THREAD_NUM buffers)
// Phase 3b: Reduce cores → yGm (core 0 reduces all per-core buffers)
//
// Index decomposition (UintDiv fast-divide, MDE §6.1):
//   idx = b × (outH×outW×depth) + h_out × (outW×depth) + w_out × depth + d
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ComputeSimt(
    int64_t totalElements, int32_t inputH, int32_t inputW, int32_t depth, int32_t filterH, int32_t filterW,
    int32_t outH, int32_t outW, int32_t strideH, int32_t strideW, int32_t rateH, int32_t rateW, int32_t padTop,
    int32_t padLeft, int32_t padInputH, int32_t padInputW, int32_t isNCHW, __ubuf__ uint64_t* uintdivUb, __gm__ T* xGm,
    __gm__ T* filterGm, __gm__ T* outBackpropGm, __gm__ T* wsPerThread, int64_t perCoreBufElems, int32_t coreId)
{
    // 1. Read UintDiv magic/shift from UB
    // NHWC order: (depth, outW, outH) → decomposes idx into (b, h_out, w_out, d)
    // NCHW order: (outW, outH, depth) → decomposes idx into (b, d, h_out, w_out)
    const uint64_t magic0 = uintdivUb[0];
    const uint64_t shift0 = uintdivUb[1];
    const uint64_t magic1 = uintdivUb[2];
    const uint64_t shift1 = uintdivUb[3];
    const uint64_t magic2 = uintdivUb[4];
    const uint64_t shift2 = uintdivUb[5];

    // 2. Pre-compute constants
    // Data access strides differ between NHWC and NCHW:
    //   NHWC: x=[N,H,W,C] in memory, filter=[fH,fW,C], y=[fH,fW,C]
    //   NCHW: x=[N,C,H,W] in memory, filter=[C,fH,fW], y=[C,fH,fW]
    const uint64_t uDepth = static_cast<uint64_t>(depth);
    const uint64_t uOutW = static_cast<uint64_t>(outW);
    const uint64_t uOutH = static_cast<uint64_t>(outH);

    // Stride computation (format-dependent)
    // xDepthStride: NHWC=1 (C contiguous), NCHW=inputH*inputW (C strided)
    // xRowStride:   NHWC=inputW*depth (H stride includes C), NCHW=inputW (H stride is just W)
    // xWStride:     NHWC=depth (W stride includes C), NCHW=1 (W contiguous)
    const int64_t xDepthStride = isNCHW ? (static_cast<int64_t>(inputH) * inputW) : 1;
    const int64_t xRowStride = isNCHW ? static_cast<int64_t>(inputW) : (static_cast<int64_t>(inputW) * depth);
    const int64_t xWStride = isNCHW ? 1 : static_cast<int64_t>(depth);
    const int64_t inputBatchStride = static_cast<int64_t>(inputH) * inputW * depth; // same for both formats

    // filterDepthStride: NHWC=1, NCHW=filterH*filterW
    // filterRowStride:   NHWC=filterW*depth, NCHW=filterW
    // filterWStride:     NHWC=depth, NCHW=1
    const int64_t filterDepthStride = isNCHW ? (static_cast<int64_t>(filterH) * filterW) : 1;
    const int64_t filterRowStride = isNCHW ? static_cast<int64_t>(filterW) : (static_cast<int64_t>(filterW) * depth);
    const int64_t filterWStride = isNCHW ? 1 : static_cast<int64_t>(depth);

    // Divisors for index decomposition (order depends on format)
    const uint64_t div0 = isNCHW ? uOutW : uDepth;
    const uint64_t div1 = isNCHW ? uOutH : uOutW;
    const uint64_t div2 = isNCHW ? uDepth : uOutH;

    // Golden alignment: argmax in type T (golden: val = x + filter, same dtype)
    const T lowestVal = GetLowestVal<T>();

    // v2.2: per-thread buffer base for this thread
    const int64_t myBufBase = (static_cast<int64_t>(coreId) * static_cast<int64_t>(THREAD_NUM) +
                               static_cast<int64_t>(threadIdx.x)) *
                              perCoreBufElems;

    // 3. Grid-Stride main loop
    const uint64_t gridStride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x);
         idx < static_cast<uint64_t>(totalElements); idx += gridStride) {
        // 3.1 Index decomposition
        // NHWC: idx → (b, h_out, w_out, d) via divisors (depth, outW, outH)
        // NCHW: idx → (b, d, h_out, w_out) via divisors (outW, outH, depth)
        uint64_t rem = idx;
        uint64_t q = Simt::UintDiv<uint64_t>(rem, magic0, shift0);
        uint64_t c0 = rem - q * div0;
        rem = q;
        q = Simt::UintDiv<uint64_t>(rem, magic1, shift1);
        uint64_t c1 = rem - q * div1;
        rem = q;
        q = Simt::UintDiv<uint64_t>(rem, magic2, shift2);
        uint64_t c2 = rem - q * div2;
        uint64_t b = q;

        uint64_t d, hOut, wOut;
        if (isNCHW) {
            wOut = c0;
            hOut = c1;
            d = c2;
        } else {
            d = c0;
            wOut = c1;
            hOut = c2;
        }

        // 3.2 Compute window start position
        int32_t hBeg = static_cast<int32_t>(hOut) * strideH - padTop;
        int32_t wBeg = static_cast<int32_t>(wOut) * strideW - padLeft;

        // 3.3 argmax search in type T (align with golden same-type addition)
        T curVal = lowestVal;
        int32_t hMax = 0;
        int32_t wMax = 0;

        const int64_t xBase = static_cast<int64_t>(b) * inputBatchStride + static_cast<int64_t>(d) * xDepthStride;
        const int64_t filterBase = static_cast<int64_t>(d) * filterDepthStride;

        for (int32_t h = 0; h < filterH; ++h) {
            int32_t hIn = hBeg + h * rateH;
            if (hIn >= 0 && hIn < padInputH) {
                int64_t xRowOffset = xBase + static_cast<int64_t>(hIn) * xRowStride;
                int64_t filterRowOffset = filterBase + static_cast<int64_t>(h) * filterRowStride;
                for (int32_t w = 0; w < filterW; ++w) {
                    int32_t wIn = wBeg + w * rateW;
                    if (wIn >= 0 && wIn < padInputW) {
                        T xVal;
                        if (hIn < inputH && wIn < inputW) {
                            xVal = xGm[xRowOffset + static_cast<int64_t>(wIn) * xWStride];
                        } else {
                            xVal = static_cast<T>(0);
                        }
                        T val = xVal + filterGm[filterRowOffset + static_cast<int64_t>(w) * filterWStride];
                        if (val > curVal) {
                            curVal = val;
                            hMax = h;
                            wMax = w;
                        }
                    }
                }
            }
        }

        // 3.4 Sequential accumulate to per-thread buffer (v2.2: no atomic_add)
        // yOffset uses format-dependent strides (same as filter strides)
        int64_t yOffset = static_cast<int64_t>(hMax) * filterRowStride + static_cast<int64_t>(wMax) * filterWStride +
                          static_cast<int64_t>(d) * filterDepthStride;
        T grad = outBackpropGm[idx];
        wsPerThread[myBufBase + yOffset] += grad;
    }
}

// ============================================================================
// LaunchComputeVf: common helper to compute UintDiv params and launch ComputeSimt
// v2.1: passes wsPerCore and perCoreBufElems to ComputeSimt (per-core buffer)
// ============================================================================
template <typename T>
__aicore__ inline void LaunchComputeVf(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm,
                                       __gm__ T* filterGm, __gm__ T* outBackpropGm, __gm__ T* wsPerThread,
                                       int32_t coreId)
{
    // Pre-compute UintDiv magic/shift (device scalar, write to UB)
    // v2.5: order depends on data_format
    //   NHWC: (depth, outW, outH) → decomposes idx into (b, h_out, w_out, d)
    //   NCHW: (outW, outH, depth) → decomposes idx into (b, d, h_out, w_out)
    LocalMemAllocator<Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> ub = ubAlloc.Alloc<uint64_t>(8);

    uint64_t magic = 0;
    uint64_t shift = 0;

    if (tilingData->isNCHW) {
        // NCHW: div0=outW, div1=outH, div2=depth
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outW));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outH));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    } else {
        // NHWC: div0=depth, div1=outW, div2=outH
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outW));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outH));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    }

    DataSyncBarrier<MemDsbT::UB>();

    __ubuf__ uint64_t* uintdivPtr = (__ubuf__ uint64_t*)ub.GetPhyAddr();

    // Launch Compute VF (v2.2: targets per-thread buffer, no atomic_add)
    // v2.5: passes isNCHW for format-dependent stride computation
    asc_vf_call<ComputeSimt<T>>(dim3(THREAD_NUM), tilingData->totalElements, tilingData->inputH, tilingData->inputW,
                                tilingData->depth, tilingData->filterH, tilingData->filterW, tilingData->outH,
                                tilingData->outW, tilingData->strideH, tilingData->strideW, tilingData->rateH,
                                tilingData->rateW, tilingData->padTop, tilingData->padLeft, tilingData->padInputH,
                                tilingData->padInputW, tilingData->isNCHW, uintdivPtr, xGm, filterGm, outBackpropGm,
                                wsPerThread, tilingData->perCoreBufElems, coreId);
}

// ============================================================================
// Phase 3a: Tree reduction of per-thread buffers → per-core buffer
// v2.4: In-place tree reduction on wsBuf, O(filterSize × log2(1024)) per core
//   - 1024 buffers → 512 → 256 → ... → 1 (10 steps)
//   - Each step: active threads merge pairs of buffers via +=
//   - asc_syncthreads() synchronizes between steps
//   - Final: buffer[0] has the sum, write to perCoreBuf
//   - Fully deterministic: fixed merge order, no atomic_add
//   - wsBuf is not used after Phase 3a, so in-place modification is safe
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ReduceThreadsSimt(int64_t filterSize,
                                                                                   int32_t needCoreNum,
                                                                                   int64_t perCoreBufElems,
                                                                                   __gm__ T* wsBuf,
                                                                                   __gm__ T* perCoreBuf, int32_t coreId)
{
    const int64_t coreBase = static_cast<int64_t>(coreId) * static_cast<int64_t>(THREAD_NUM) * perCoreBufElems;
    const int64_t outBase = static_cast<int64_t>(coreId) * perCoreBufElems;

    const uint32_t tid = threadIdx.x;
    const uint64_t end = static_cast<uint64_t>(filterSize);

    // Tree reduction: 1024 → 512 → 256 → ... → 1 (10 steps)
    // Each active thread merges buffer[tid + offset] into buffer[tid]
    for (uint32_t offset = THREAD_NUM / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int64_t dstOffset = coreBase + static_cast<int64_t>(tid) * perCoreBufElems;
            int64_t srcOffset = coreBase + static_cast<int64_t>(tid + offset) * perCoreBufElems;
            for (uint64_t i = 0; i < end; ++i) {
                wsBuf[dstOffset + static_cast<int64_t>(i)] += wsBuf[srcOffset + static_cast<int64_t>(i)];
            }
        }
        asc_syncthreads();
    }

    // After tree reduction, buffer[0] has the sum
    // Write to perCoreBuf
    if (tid == 0) {
        int64_t srcOffset = coreBase;
        for (uint64_t i = 0; i < end; ++i) {
            perCoreBuf[outBase + static_cast<int64_t>(i)] = wsBuf[srcOffset + static_cast<int64_t>(i)];
        }
    }
}

// ============================================================================
// Phase 3b: Final Reduce — sequentially accumulate per-core buffers to yGm
//
// v2.2: Only core 0 executes this VF (controlled in Process scalar scope)
// For each filter element i, iterate cores in order 0→1→...→needCoreNum-1
// and accumulate perCoreBuf[core * perCoreBufElems + i] to yGm[i].
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ReduceSimt(int64_t filterSize, int32_t needCoreNum,
                                                                            int64_t perCoreBufElems,
                                                                            __gm__ T* perCoreBuf, __gm__ T* yGm)
{
    // Phase 3b is launched only by core 0 (controlled in Process scalar scope)
    const uint64_t totalThreads = static_cast<uint64_t>(blockDim.x);
    const uint64_t end = static_cast<uint64_t>(filterSize);

    for (uint64_t idx = static_cast<uint64_t>(threadIdx.x); idx < end; idx += totalThreads) {
        // Sequential accumulation across cores (deterministic order)
        T sum = static_cast<T>(0);
        for (int32_t core = 0; core < needCoreNum; ++core) {
            sum += perCoreBuf[static_cast<int64_t>(core) * perCoreBufElems + static_cast<int64_t>(idx)];
        }
        yGm[idx] = sum;
    }
}

// ============================================================================
// Process: four-phase fully deterministic dispatcher
//
// v2.4: SyncAll 8→2, removed redundant zero-out, tree reduction in Phase 3a
//   Phase 1  - ZeroOut: grid-stride zero-fill per-thread workspace
//   Phase 2  - Compute: argmax + sequential += to per-thread buffer (no atomic_add)
//   Phase 3a - ReduceThreads: tree reduction (1024→1, O(10×filterSize))
//   Phase 3b - ReduceCores: core 0 reduces all per-core buffers → yGm
//
// SyncAll: only 2 (Phase 1→2 cross-core, Phase 3a→3b cross-core)
// Workspace layout: wsBuf[needCoreNum × THREAD_NUM × perCoreBufElems] + perCoreBuf[needCoreNum × perCoreBufElems]
// ============================================================================
template <typename T>
__aicore__ inline void Process(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm, __gm__ T* filterGm,
                               __gm__ T* outBackpropGm, __gm__ T* yGm, __gm__ T* wsBuf)
{
    int32_t coreId = static_cast<int32_t>(GetBlockIdx());

    // ===== Empty tensor fast path =====
    if (tilingData->totalElements == 0 || tilingData->filterSize == 0) {
        asc_vf_call<ZeroOutSimt<T>>(dim3(THREAD_NUM), tilingData->filterSize, yGm);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return;
    }

    // ===== Phase 1: grid-stride zero-out per-thread workspace =====
    // v2.4: yGm zero-out removed — Phase 3b writes all yGm elements with `=`.
    int64_t wsTotalElems = static_cast<int64_t>(tilingData->needCoreNum) * static_cast<int64_t>(THREAD_NUM) *
                           tilingData->perCoreBufElems;
    asc_vf_call<ZeroOutSimt<T>>(dim3(THREAD_NUM), wsTotalElems, wsBuf);

    GlobalTensor<T> wsTensor;
    wsTensor.SetGlobalBuffer(wsBuf, static_cast<uint64_t>(wsTotalElems));
    SyncAll();

    // ===== Phase 2: compute argmax + sequential += to per-thread buffer =====
    LaunchComputeVf<T>(tilingData, xGm, filterGm, outBackpropGm, wsBuf, coreId);

    // ===== Phase 3a: tree reduction of per-thread buffers → per-core buffer =====
    // v2.4: perCoreBuf zero-out removed — Phase 3a writes all elements with `=`.
    int64_t perCoreBufOffset = static_cast<int64_t>(tilingData->needCoreNum) * static_cast<int64_t>(THREAD_NUM) *
                               tilingData->perCoreBufElems;
    __gm__ T* perCoreBuf = wsBuf + perCoreBufOffset;

    asc_vf_call<ReduceThreadsSimt<T>>(dim3(THREAD_NUM), tilingData->filterSize, tilingData->needCoreNum,
                                      tilingData->perCoreBufElems, wsBuf, perCoreBuf, coreId);

    // SyncAll (cross-core: Phase 3b reads ALL cores' perCoreBuf)
    int64_t perCoreBufTotal = static_cast<int64_t>(tilingData->needCoreNum) * tilingData->perCoreBufElems;
    GlobalTensor<T> pcTensor;
    pcTensor.SetGlobalBuffer(perCoreBuf, static_cast<uint64_t>(perCoreBufTotal));
    SyncAll();

    // ===== Phase 3b: final reduce (core 0 only) =====
    if (coreId == 0) {
        asc_vf_call<ReduceSimt<T>>(dim3(THREAD_NUM), tilingData->filterSize, tilingData->needCoreNum,
                                   tilingData->perCoreBufElems, perCoreBuf, yGm);

        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
    }
}

} // namespace NsDilation2DBackpropFilter
#endif // DILATION2D_BACKPROP_FILTER_SIMT_H_
