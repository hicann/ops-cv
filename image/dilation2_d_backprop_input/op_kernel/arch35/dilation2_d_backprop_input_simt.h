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
 * \file dilation2_d_backprop_input_simt.h
 * \brief SIMT kernel: SAME padding uses replicate (clamp to edge) + Phase 2.5 edge scan
 *        VALID/CALCULATED: skip OOB positions (no replicate)
 *        Auto-detect: when window > input, enable edge scan for SAME padding
 */

#ifndef DILATION2_D_BACKPROP_INPUT_SIMT_H_
#define DILATION2_D_BACKPROP_INPUT_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/cpp/kernel_simt_math_intf.h"
#include "dilation2_d_backprop_input_tiling_data.h"
#include "dilation2_d_backprop_input_tiling_key.h"
#include <limits>

namespace NsDilation2DBackpropInput {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
constexpr uint32_t UINTDIV_PARAM_COUNT = 8;
constexpr int32_t CLAMPED_ARGMAX_OFFSET = 2;
constexpr int32_t PADDING_MODE_SAME = 0;
constexpr int32_t PADDING_MODE_VALID = 1;
constexpr int32_t PADDING_MODE_CALCULATED = 2;
constexpr float
    ARGMAX_LOWEST = -std::numeric_limits<float>::max(); // -FLT_MAX, same as TF Eigen::NumTraits<float>::lowest()

template <uint32_t schMode>
__simt_callee__ inline int64_t ComputeInIdx(int64_t b, int64_t hInMax, int64_t wInMax, int64_t d, int64_t inputH,
                                            int64_t inputW, int64_t depth)
{
    if constexpr (schMode == 0) {
        return b * inputH * inputW * depth + hInMax * inputW * depth + wInMax * depth + d;
    } else {
        return b * depth * inputH * inputW + d * inputH * inputW + hInMax * inputW + wInMax;
    }
}
// argmax: align with TF CPU implementation (dilation_ops.cc).
// SAME/VALID: skip OOB, init h_in_max=max(0,h_beg). If no valid position beats curVal, use clamped edge.
// CALCULATED: OOB positions have val=-FLT_MAX+filter (simulating -FLT_MAX padding), compete with in-bounds.
//   If argmax lands on OOB, gradient dropped (TF slices back).
template <typename T, uint32_t schMode>
__simt_callee__ inline void FindArgmaxReplicate(int64_t hBeg, int64_t wBeg, int64_t inputH, int64_t inputW,
                                                int64_t depth, int64_t filterH, int64_t filterW, int64_t rateH,
                                                int64_t rateW, int64_t b, int64_t d, __gm__ T* xGm, __gm__ T* filterGm,
                                                int64_t& hInMax, int64_t& wInMax, bool& isClamped, int32_t paddingMode)
{
    float curVal = ARGMAX_LOWEST;
    hInMax = -1;
    wInMax = -1;
    isClamped = false;
    bool isCalculated = (paddingMode == PADDING_MODE_CALCULATED);
    bool foundValid = false;
    bool oobArgmax = false;
    for (int64_t fh = 0; fh < filterH; fh++) {
        int64_t hIn = hBeg + fh * rateH;
        bool hValid = (hIn >= 0 && hIn < inputH);
        for (int64_t fw = 0; fw < filterW; fw++) {
            int64_t wIn = wBeg + fw * rateW;
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
            float val = inBounds ?
                            (schMode == 0 ?
                                 static_cast<float>(
                                     xGm[b * inputH * inputW * depth + hIn * inputW * depth + wIn * depth + d]) +
                                     filterVal :
                                 static_cast<float>(
                                     xGm[b * depth * inputH * inputW + d * inputH * inputW + hIn * inputW + wIn]) +
                                     filterVal) :
                            ARGMAX_LOWEST + filterVal;
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
    if (oobArgmax) {
        hInMax = -1;
        wInMax = -1;
    } else if (!foundValid) {
        int64_t hInit = (hBeg < 0) ? 0 : hBeg;
        int64_t wInit = (wBeg < 0) ? 0 : wBeg;
        hInMax = hInit;
        wInMax = wInit;
        bool inBounds = (hInit >= 0 && hInit < inputH && wInit >= 0 && wInit < inputW);
        if (paddingMode == PADDING_MODE_SAME) {
            isClamped = (hBeg < 0 || wBeg < 0) && inBounds;
            if (!inBounds) {
                hInMax = -1;
                wInMax = -1;
            }
        } else {
            // CALCULATED/VALID: keep only if (hBeg,wBeg) is in-bounds (reverse-mappable)
            if (!(hBeg >= 0 && wBeg >= 0 && inBounds)) {
                hInMax = -1;
                wInMax = -1;
            }
        }
    }
}

// Phase 2 accumulation + edge scan for clamped argmax.
template <typename T, uint32_t schMode>
__simt_callee__ inline float AccumulateWithEdgeScan(int64_t b, int64_t hIn, int64_t wIn, int64_t d, int64_t inputH,
                                                    int64_t inputW, int64_t depth, int64_t outputH, int64_t outputW,
                                                    int64_t filterH, int64_t filterW, int64_t strideH, int64_t strideW,
                                                    int64_t rateH, int64_t rateW, int64_t padTop, int64_t padLeft,
                                                    __gm__ T* outBackpropGm, __gm__ int64_t* argmaxWorkspace,
                                                    int64_t inIdx, bool needEdgeScan, int64_t hOutMax, int64_t wOutMax)
{
    float sum = 0.0f;
    for (int64_t fh = 0; fh < filterH; fh++) {
        int64_t numH = hIn + padTop - fh * rateH;
        if (numH >= 0 && numH % strideH == 0) {
            int64_t hOut = numH / strideH;
            if (hOut < outputH) {
                for (int64_t fw = 0; fw < filterW; fw++) {
                    int64_t numW = wIn + padLeft - fw * rateW;
                    if (numW >= 0 && numW % strideW == 0) {
                        int64_t wOut = numW / strideW;
                        if (wOut < outputW) {
                            int64_t outIdx;
                            if constexpr (schMode == 0) {
                                outIdx = b * outputH * outputW * depth + hOut * outputW * depth + wOut * depth + d;
                            } else {
                                outIdx = b * depth * outputH * outputW + d * outputH * outputW + hOut * outputW + wOut;
                            }
                            if (argmaxWorkspace[outIdx] == inIdx) {
                                sum += static_cast<float>(outBackpropGm[outIdx]);
                            }
                        }
                    }
                }
            }
        }
    }
    if (needEdgeScan && (hIn == 0 || hIn == inputH - 1 || wIn == 0 || wIn == inputW - 1)) {
        int64_t outIdxBase;
        if constexpr (schMode == 0) {
            outIdxBase = b * outputH * outputW * depth + d;
        } else {
            outIdxBase = b * depth * outputH * outputW + d * outputH * outputW;
        }
        int64_t rowStride = (schMode == 0) ? outputW * depth : outputW;
        for (int64_t hOut = 0; hOut < outputH; hOut++) {
            int64_t wOutEnd = (hOut < hOutMax) ? outputW : wOutMax;
            if (wOutEnd <= 0)
                break;
            int64_t outIdx = outIdxBase + hOut * rowStride;
            for (int64_t wOut = 0; wOut < wOutEnd; wOut++) {
                int64_t argmaxVal = argmaxWorkspace[outIdx];
                if (argmaxVal < -1 && (-argmaxVal - CLAMPED_ARGMAX_OFFSET) == inIdx) {
                    sum += static_cast<float>(outBackpropGm[outIdx]);
                }
                outIdx += (schMode == 0) ? depth : 1;
            }
        }
    }
    return sum;
}

// ========== Phase 1 ==========
template <typename T, uint32_t schMode>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void Phase1ArgmaxKernel(
    int64_t outTotalNum, int64_t batch, int64_t inputH, int64_t inputW, int64_t depth, int64_t outputH, int64_t outputW,
    int64_t filterH, int64_t filterW, int64_t strideH, int64_t strideW, int64_t rateH, int64_t rateW, int64_t padTop,
    int64_t padLeft, __ubuf__ uint64_t* uintdivUb, __gm__ T* xGm, __gm__ T* filterGm,
    __gm__ volatile int64_t* argmaxWorkspace, int32_t paddingMode)
{
    const uint64_t magic0 = uintdivUb[0];
    const uint64_t shift0 = uintdivUb[1];
    const uint64_t magic1 = uintdivUb[2];
    const uint64_t shift1 = uintdivUb[3];
    const uint64_t magic2 = uintdivUb[4];
    const uint64_t shift2 = uintdivUb[5];
    const uint64_t uDepth = static_cast<uint64_t>(depth);
    const uint64_t uOutW = static_cast<uint64_t>(outputW);
    const uint64_t uOutH = static_cast<uint64_t>(outputH);
    const uint64_t div0 = (schMode == 0) ? uDepth : uOutW;
    const uint64_t div1 = (schMode == 0) ? uOutW : uOutH;
    const uint64_t div2 = (schMode == 0) ? uOutH : uDepth;
    const uint64_t gridStride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < static_cast<uint64_t>(outTotalNum); idx += gridStride) {
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
        int64_t bInt, hOut, wOut, d;
        if constexpr (schMode == 0) {
            d = static_cast<int64_t>(c0);
            wOut = static_cast<int64_t>(c1);
            hOut = static_cast<int64_t>(c2);
        } else {
            wOut = static_cast<int64_t>(c0);
            hOut = static_cast<int64_t>(c1);
            d = static_cast<int64_t>(c2);
        }
        bInt = static_cast<int64_t>(b);

        int64_t hBeg = hOut * strideH - padTop;
        int64_t wBeg = wOut * strideW - padLeft;
        int64_t hInMax, wInMax;
        bool isClamped;
        FindArgmaxReplicate<T, schMode>(hBeg, wBeg, inputH, inputW, depth, filterH, filterW, rateH, rateW, bInt, d, xGm,
                                        filterGm, hInMax, wInMax, isClamped, paddingMode);
        if (hInMax < 0) {
            argmaxWorkspace[idx] = -1;
        } else {
            int64_t inIdx = ComputeInIdx<schMode>(bInt, hInMax, wInMax, d, inputH, inputW, depth);
            argmaxWorkspace[idx] = isClamped ? (-inIdx - CLAMPED_ARGMAX_OFFSET) : inIdx;
        }
    }
}

// ========== Phase 2 ==========
template <typename T, uint32_t schMode>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void Phase2AccumulateKernel(
    int64_t inTotalNum, int64_t batch, int64_t inputH, int64_t inputW, int64_t depth, int64_t outputH, int64_t outputW,
    int64_t filterH, int64_t filterW, int64_t strideH, int64_t strideW, int64_t rateH, int64_t rateW, int64_t padTop,
    int64_t padLeft, __ubuf__ uint64_t* uintdivUb, __gm__ T* outBackpropGm, __gm__ int64_t* argmaxWorkspace,
    __gm__ T* inBackpropGm, int32_t paddingMode)
{
    bool needEdgeScan = (padTop > 0 || padLeft > 0);
    int64_t hOutMax = needEdgeScan ? ((padTop > 0) ? (padTop + strideH - 1) / strideH : 0) : 0;
    int64_t wOutMax = needEdgeScan ? ((padLeft > 0) ? (padLeft + strideW - 1) / strideW : 0) : 0;
    hOutMax = (hOutMax < outputH) ? hOutMax : outputH;
    wOutMax = (wOutMax < outputW) ? wOutMax : outputW;
    if (hOutMax == 0 && wOutMax == 0)
        needEdgeScan = false;

    const uint64_t magic0 = uintdivUb[0];
    const uint64_t shift0 = uintdivUb[1];
    const uint64_t magic1 = uintdivUb[2];
    const uint64_t shift1 = uintdivUb[3];
    const uint64_t magic2 = uintdivUb[4];
    const uint64_t shift2 = uintdivUb[5];
    const uint64_t uDepth = static_cast<uint64_t>(depth);
    const uint64_t uInW = static_cast<uint64_t>(inputW);
    const uint64_t uInH = static_cast<uint64_t>(inputH);
    const uint64_t div0 = (schMode == 0) ? uDepth : uInW;
    const uint64_t div1 = (schMode == 0) ? uInW : uInH;
    const uint64_t div2 = (schMode == 0) ? uInH : uDepth;
    const uint64_t gridStride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < static_cast<uint64_t>(inTotalNum); idx += gridStride) {
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
        int64_t bInt, hIn, wIn, d;
        if constexpr (schMode == 0) {
            d = static_cast<int64_t>(c0);
            wIn = static_cast<int64_t>(c1);
            hIn = static_cast<int64_t>(c2);
        } else {
            wIn = static_cast<int64_t>(c0);
            hIn = static_cast<int64_t>(c1);
            d = static_cast<int64_t>(c2);
        }
        bInt = static_cast<int64_t>(b);

        float sum = AccumulateWithEdgeScan<T, schMode>(bInt, hIn, wIn, d, inputH, inputW, depth, outputH, outputW,
                                                       filterH, filterW, strideH, strideW, rateH, rateW, padTop,
                                                       padLeft, outBackpropGm, argmaxWorkspace,
                                                       static_cast<int64_t>(idx), needEdgeScan, hOutMax, wOutMax);
        inBackpropGm[idx] = static_cast<T>(sum);
    }
}

// ========== Process ==========
template <typename T, uint32_t schMode>
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
    __gm__ int64_t* argmaxWorkspace = (__gm__ int64_t*)GetUserWorkspace(workspace);
    if (argmaxWorkspace == nullptr) {
        return;
    }

    int32_t paddingMode = tilingData->paddingMode;
    uint64_t magic = 0;
    uint64_t shift = 0;

    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> ub1 = ubAlloc.Alloc<uint64_t>(UINTDIV_PARAM_COUNT);
    if constexpr (schMode == 0) {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub1.SetValue(0, magic);
        ub1.SetValue(1, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outputW));
        ub1.SetValue(2, magic);
        ub1.SetValue(3, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outputH));
        ub1.SetValue(4, magic);
        ub1.SetValue(5, shift);
    } else {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outputW));
        ub1.SetValue(0, magic);
        ub1.SetValue(1, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outputH));
        ub1.SetValue(2, magic);
        ub1.SetValue(3, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub1.SetValue(4, magic);
        ub1.SetValue(5, shift);
    }

    asc_vf_call<Phase1ArgmaxKernel<T, schMode>>(
        dim3(THREAD_NUM), tilingData->outTotalNum, tilingData->batch, tilingData->inputH, tilingData->inputW,
        tilingData->depth, tilingData->outputH, tilingData->outputW, tilingData->filterH, tilingData->filterW,
        tilingData->strideH, tilingData->strideW, tilingData->rateH, tilingData->rateW, tilingData->padTop,
        tilingData->padLeft, (__ubuf__ uint64_t*)ub1.GetPhyAddr(), xGm, filterGm, argmaxWorkspace, paddingMode);

    SyncAll();

    LocalTensor<uint64_t> ub2 = ubAlloc.Alloc<uint64_t>(UINTDIV_PARAM_COUNT);
    if constexpr (schMode == 0) {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub2.SetValue(0, magic);
        ub2.SetValue(1, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->inputW));
        ub2.SetValue(2, magic);
        ub2.SetValue(3, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->inputH));
        ub2.SetValue(4, magic);
        ub2.SetValue(5, shift);
    } else {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->inputW));
        ub2.SetValue(0, magic);
        ub2.SetValue(1, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->inputH));
        ub2.SetValue(2, magic);
        ub2.SetValue(3, shift);
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub2.SetValue(4, magic);
        ub2.SetValue(5, shift);
    }

    asc_vf_call<Phase2AccumulateKernel<T, schMode>>(
        dim3(THREAD_NUM), tilingData->inTotalNum, tilingData->batch, tilingData->inputH, tilingData->inputW,
        tilingData->depth, tilingData->outputH, tilingData->outputW, tilingData->filterH, tilingData->filterW,
        tilingData->strideH, tilingData->strideW, tilingData->rateH, tilingData->rateW, tilingData->padTop,
        tilingData->padLeft, (__ubuf__ uint64_t*)ub2.GetPhyAddr(), outBackpropGm, argmaxWorkspace, inBackpropGm,
        paddingMode);

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace NsDilation2DBackpropInput
#endif // DILATION2_D_BACKPROP_INPUT_SIMT_H_
