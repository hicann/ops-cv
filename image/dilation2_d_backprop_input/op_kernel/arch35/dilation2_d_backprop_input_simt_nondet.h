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
 * \file dilation2_d_backprop_input_simt_nondet.h
 * \brief SIMT kernel implementation for dilation2_d_backprop_input (non-deterministic)
 */

#ifndef DILATION2_D_BACKPROP_INPUT_SIMT_NONDET_H_
#define DILATION2_D_BACKPROP_INPUT_SIMT_NONDET_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/cpp/kernel_simt_math_intf.h"
#include "dilation2_d_backprop_input_tiling_data.h"
#include "dilation2_d_backprop_input_tiling_key.h"
#include <limits>
#include <cstdint>
#include <type_traits>

namespace NsDilation2DBackpropInputNonDet {
using namespace AscendC;

// ========== Common constants ==========
constexpr uint32_t THREAD_NUM = 512;
constexpr uint32_t UINTDIV_PARAM_COUNT = 8;
constexpr int32_t PADDING_MODE_SAME = 0;
constexpr int32_t PADDING_MODE_VALID = 1;
constexpr int32_t PADDING_MODE_CALCULATED = 2;
constexpr float ARGMAX_LOWEST = -std::numeric_limits<float>::max();

// ========== Non-deterministic constants ==========
constexpr uint32_t DIMS_PARAM_COUNT = 16;

constexpr uint32_t IDX_BATCH = 0;
constexpr uint32_t IDX_INPUT_H = 1;
constexpr uint32_t IDX_INPUT_W = 2;
constexpr uint32_t IDX_DEPTH = 3;
constexpr uint32_t IDX_OUTPUT_H = 4;
constexpr uint32_t IDX_OUTPUT_W = 5;
constexpr uint32_t IDX_FILTER_H = 6;
constexpr uint32_t IDX_FILTER_W = 7;
constexpr uint32_t IDX_STRIDE_H = 8;
constexpr uint32_t IDX_STRIDE_W = 9;
constexpr uint32_t IDX_RATE_H = 10;
constexpr uint32_t IDX_RATE_W = 11;
constexpr uint32_t IDX_PAD_TOP = 12;
constexpr uint32_t IDX_PAD_LEFT = 13;
constexpr uint32_t IDX_PADDING_MODE = 14;
constexpr uint32_t IDX_IN_TOTAL_NUM = 15;

// ========== ComputeInIdx ==========
// IdxT: index width (int32_t small-shape fast path / int64_t general path).
template <uint32_t schMode, typename IdxT = int64_t>
__simt_callee__ inline IdxT ComputeInIdx(IdxT b, IdxT hInMax, IdxT wInMax, IdxT d, IdxT inputH, IdxT inputW, IdxT depth)
{
    if constexpr (schMode == 0) {
        return b * inputH * inputW * depth + hInMax * inputW * depth + wInMax * depth + d;
    } else {
        return b * depth * inputH * inputW + d * inputH * inputW + hInMax * inputW + wInMax;
    }
}

// ========== DecomposeOutIndex ==========
template <uint32_t schMode, typename IdxT = int64_t>
__simt_callee__ inline void DecomposeOutIndex(typename std::make_unsigned<IdxT>::type idx, __ubuf__ uint64_t* uintdivUb,
                                              IdxT& bInt, IdxT& hOut, IdxT& wOut, IdxT& d, IdxT depth, IdxT outputH,
                                              IdxT outputW)
{
    using UIdxT = typename std::make_unsigned<IdxT>::type;
    const UIdxT magic0 = static_cast<UIdxT>(uintdivUb[0]);
    const UIdxT shift0 = static_cast<UIdxT>(uintdivUb[1]);
    const UIdxT magic1 = static_cast<UIdxT>(uintdivUb[2]);
    const UIdxT shift1 = static_cast<UIdxT>(uintdivUb[3]);
    const UIdxT magic2 = static_cast<UIdxT>(uintdivUb[4]);
    const UIdxT shift2 = static_cast<UIdxT>(uintdivUb[5]);
    UIdxT div0, div1, div2;
    if constexpr (schMode == 0) {
        div0 = static_cast<UIdxT>(depth);
        div1 = static_cast<UIdxT>(outputW);
        div2 = static_cast<UIdxT>(outputH);
    } else {
        div0 = static_cast<UIdxT>(outputW);
        div1 = static_cast<UIdxT>(outputH);
        div2 = static_cast<UIdxT>(depth);
    }
    UIdxT rem = idx;
    UIdxT q = Simt::UintDiv<UIdxT>(rem, magic0, shift0);
    UIdxT c0 = rem - q * div0;
    rem = q;
    q = Simt::UintDiv<UIdxT>(rem, magic1, shift1);
    UIdxT c1 = rem - q * div1;
    rem = q;
    q = Simt::UintDiv<UIdxT>(rem, magic2, shift2);
    UIdxT c2 = rem - q * div2;
    if constexpr (schMode == 0) {
        d = static_cast<IdxT>(c0);
        wOut = static_cast<IdxT>(c1);
        hOut = static_cast<IdxT>(c2);
    } else {
        wOut = static_cast<IdxT>(c0);
        hOut = static_cast<IdxT>(c1);
        d = static_cast<IdxT>(c2);
    }
    bInt = static_cast<IdxT>(q);
}

// ========== ResolveArgmaxClamp ==========
template <typename IdxT = int64_t>
__simt_callee__ inline void ResolveArgmaxClamp(IdxT hBeg, IdxT wBeg, IdxT inputH, IdxT inputW, bool foundValid,
                                               bool oobArgmax, int32_t paddingMode, IdxT& hInMax, IdxT& wInMax)
{
    if (oobArgmax) {
        hInMax = -1;
        wInMax = -1;
        return;
    }
    if (foundValid) {
        return;
    }
    if (paddingMode == PADDING_MODE_SAME) {
        hInMax = (hBeg < 0) ? 0 : hBeg;
        wInMax = (wBeg < 0) ? 0 : wBeg;
    } else {
        hInMax = hBeg;
        wInMax = wBeg;
    }
    if (hInMax < 0 || wInMax < 0 || hInMax >= inputH || wInMax >= inputW) {
        hInMax = -1;
        wInMax = -1;
    }
}

// ========== FindArgmaxNonDet ==========
template <typename T, uint32_t schMode, typename IdxT = int64_t>
__simt_callee__ inline void FindArgmaxNonDet(IdxT hBeg, IdxT wBeg, IdxT inputH, IdxT inputW, IdxT depth, IdxT filterH,
                                             IdxT filterW, IdxT rateH, IdxT rateW, IdxT b, IdxT d, __gm__ T* xGm,
                                             __gm__ T* filterGm, IdxT& hInMax, IdxT& wInMax, int32_t paddingMode)
{
    float curVal = ARGMAX_LOWEST;
    hInMax = -1;
    wInMax = -1;
    bool isCalculated = (paddingMode == PADDING_MODE_CALCULATED);
    bool foundValid = false;
    bool oobArgmax = false;
    for (IdxT fh = 0; fh < filterH; fh++) {
        IdxT hIn = hBeg + fh * rateH;
        bool hValid = (hIn >= 0 && hIn < inputH);
        for (IdxT fw = 0; fw < filterW; fw++) {
            IdxT wIn = wBeg + fw * rateW;
            bool wValid = (wIn >= 0 && wIn < inputW);
            bool inBounds = (hValid && wValid);
            if (!inBounds && !isCalculated) {
                continue;
            }
            float filterVal;
            if constexpr (schMode == 0) {
                filterVal = static_cast<float>(filterGm[fh * filterW * depth + fw * depth + d]);
            } else {
                filterVal = static_cast<float>(filterGm[d * filterH * filterW + fh * filterW + fw]);
            }
            // Raw values, aligned with TF CPU: no NaN/Inf normalization, NaN/-Inf never win (strict >)
            float xVal = inBounds ? static_cast<float>(
                                        xGm[ComputeInIdx<schMode, IdxT>(b, hIn, wIn, d, inputH, inputW, depth)]) :
                                    0.0f;
            float val = inBounds ? (xVal + filterVal) : (ARGMAX_LOWEST + filterVal);
            if (val > curVal) {
                curVal = val;
                if (inBounds) {
                    hInMax = hIn;
                    wInMax = wIn;
                    foundValid = true;
                    oobArgmax = false;
                } else {
                    oobArgmax = true;
                }
            }
        }
    }
    ResolveArgmaxClamp<IdxT>(hBeg, wBeg, inputH, inputW, foundValid, oobArgmax, paddingMode, hInMax, wInMax);
}

// ========== ZeroOutSimt ==========
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ZeroOutSimt(int64_t inTotalNum, __gm__ T* inBackpropGm)
{
    const uint64_t gridStride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    const uint64_t end = static_cast<uint64_t>(inTotalNum);
    for (uint64_t idx =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x);
         idx < end; idx += gridStride) {
        inBackpropGm[idx] = static_cast<T>(0);
    }
}

// ========== BackpropKernel ==========
template <typename T, uint32_t schMode, typename IdxT = int64_t>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void BackpropKernel(
    int64_t outTotalNum, __ubuf__ int64_t* dimsUb, __ubuf__ uint64_t* uintdivUb, __gm__ T* xGm, __gm__ T* filterGm,
    __gm__ T* outBackpropGm, __gm__ T* inBackpropGm)
{
    const IdxT batch = static_cast<IdxT>(dimsUb[IDX_BATCH]);
    const IdxT inputH = static_cast<IdxT>(dimsUb[IDX_INPUT_H]);
    const IdxT inputW = static_cast<IdxT>(dimsUb[IDX_INPUT_W]);
    const IdxT depth = static_cast<IdxT>(dimsUb[IDX_DEPTH]);
    const IdxT outputH = static_cast<IdxT>(dimsUb[IDX_OUTPUT_H]);
    const IdxT outputW = static_cast<IdxT>(dimsUb[IDX_OUTPUT_W]);
    const IdxT filterH = static_cast<IdxT>(dimsUb[IDX_FILTER_H]);
    const IdxT filterW = static_cast<IdxT>(dimsUb[IDX_FILTER_W]);
    const IdxT strideH = static_cast<IdxT>(dimsUb[IDX_STRIDE_H]);
    const IdxT strideW = static_cast<IdxT>(dimsUb[IDX_STRIDE_W]);
    const IdxT rateH = static_cast<IdxT>(dimsUb[IDX_RATE_H]);
    const IdxT rateW = static_cast<IdxT>(dimsUb[IDX_RATE_W]);
    const IdxT padTop = static_cast<IdxT>(dimsUb[IDX_PAD_TOP]);
    const IdxT padLeft = static_cast<IdxT>(dimsUb[IDX_PAD_LEFT]);
    const int32_t paddingMode = static_cast<int32_t>(dimsUb[IDX_PADDING_MODE]);

    using UIdxT = typename std::make_unsigned<IdxT>::type;
    const UIdxT gridStride = static_cast<UIdxT>(blockDim.x) * static_cast<UIdxT>(gridDim.x);
    const UIdxT end = static_cast<UIdxT>(outTotalNum);
    for (UIdxT idx = static_cast<UIdxT>(blockIdx.x) * static_cast<UIdxT>(blockDim.x) + static_cast<UIdxT>(threadIdx.x);
         idx < end; idx += gridStride) {
        IdxT bInt, hOut, wOut, d;
        DecomposeOutIndex<schMode, IdxT>(idx, uintdivUb, bInt, hOut, wOut, d, depth, outputH, outputW);
        IdxT hBeg = hOut * strideH - padTop;
        IdxT wBeg = wOut * strideW - padLeft;
        IdxT hInMax, wInMax;
        FindArgmaxNonDet<T, schMode, IdxT>(hBeg, wBeg, inputH, inputW, depth, filterH, filterW, rateH, rateW, bInt, d,
                                           xGm, filterGm, hInMax, wInMax, paddingMode);
        if (hInMax >= 0) {
            IdxT inIdx = ComputeInIdx<schMode, IdxT>(bInt, hInMax, wInMax, d, inputH, inputW, depth);
            // Raw gradient, aligned with TF CPU: Inf/NaN propagate as-is (no clamping)
            float gradVal = static_cast<float>(outBackpropGm[idx]);
            asc_atomic_add(inBackpropGm + inIdx, gradVal);
        }
    }
}

// ========== Process (non-deterministic) ==========
template <typename T, uint32_t schMode, typename IdxT = int64_t>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workspace,
                               GM_ADDR tiling, const Dilation2DBackpropInputTilingData* tilingData)
{
    if (tilingData->inTotalNum == 0) {
        return;
    }
    __gm__ T* xGm = (__gm__ T*)x;
    __gm__ T* filterGm = (__gm__ T*)filter;
    __gm__ T* outBackpropGm = (__gm__ T*)out_backprop;
    __gm__ T* inBackpropGm = (__gm__ T*)y;

    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> ub1 = ubAlloc.Alloc<uint64_t>(UINTDIV_PARAM_COUNT);
    LocalTensor<int64_t> ub2 = ubAlloc.Alloc<int64_t>(DIMS_PARAM_COUNT);

    using UIdxT = typename std::make_unsigned<IdxT>::type;
    UIdxT magic = 0;
    UIdxT shift = 0;
    if constexpr (schMode == 0) {
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->depth));
        ub1.SetValue(0, magic);
        ub1.SetValue(1, shift);
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->outputW));
        ub1.SetValue(2, magic);
        ub1.SetValue(3, shift);
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->outputH));
        ub1.SetValue(4, magic);
        ub1.SetValue(5, shift);
    } else {
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->outputW));
        ub1.SetValue(0, magic);
        ub1.SetValue(1, shift);
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->outputH));
        ub1.SetValue(2, magic);
        ub1.SetValue(3, shift);
        GetUintDivMagicAndShift<UIdxT>(magic, shift, static_cast<UIdxT>(tilingData->depth));
        ub1.SetValue(4, magic);
        ub1.SetValue(5, shift);
    }

    ub2.SetValue(IDX_BATCH, tilingData->batch);
    ub2.SetValue(IDX_INPUT_H, tilingData->inputH);
    ub2.SetValue(IDX_INPUT_W, tilingData->inputW);
    ub2.SetValue(IDX_DEPTH, tilingData->depth);
    ub2.SetValue(IDX_OUTPUT_H, tilingData->outputH);
    ub2.SetValue(IDX_OUTPUT_W, tilingData->outputW);
    ub2.SetValue(IDX_FILTER_H, tilingData->filterH);
    ub2.SetValue(IDX_FILTER_W, tilingData->filterW);
    ub2.SetValue(IDX_STRIDE_H, tilingData->strideH);
    ub2.SetValue(IDX_STRIDE_W, tilingData->strideW);
    ub2.SetValue(IDX_RATE_H, tilingData->rateH);
    ub2.SetValue(IDX_RATE_W, tilingData->rateW);
    ub2.SetValue(IDX_PAD_TOP, tilingData->padTop);
    ub2.SetValue(IDX_PAD_LEFT, tilingData->padLeft);
    ub2.SetValue(IDX_PADDING_MODE, tilingData->paddingMode);
    ub2.SetValue(IDX_IN_TOTAL_NUM, tilingData->inTotalNum);

    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);

    asc_vf_call<ZeroOutSimt<T>>(dim3(THREAD_NUM), tilingData->inTotalNum, inBackpropGm);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    GlobalTensor<T> inBackpropDcache;
    inBackpropDcache.SetGlobalBuffer(inBackpropGm, static_cast<uint64_t>(tilingData->inTotalNum));
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(inBackpropDcache);

    SyncAll();

    asc_vf_call<BackpropKernel<T, schMode, IdxT>>(
        dim3(THREAD_NUM), tilingData->outTotalNum, (__ubuf__ int64_t*)ub2.GetPhyAddr(),
        (__ubuf__ uint64_t*)ub1.GetPhyAddr(), xGm, filterGm, outBackpropGm, inBackpropGm);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace NsDilation2DBackpropInputNonDet
#endif // DILATION2_D_BACKPROP_INPUT_SIMT_NONDET_H_
