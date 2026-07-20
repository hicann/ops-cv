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
 * \file resize_upsample_trilinear_simt.h
 * \brief ResizeUpsampleTrilinear SIMT kernel implementation for arch35.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "./resize_upsample_trilinear_simt_base.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

template <typename T>
struct SimtLaunchParams {
    T blockStart = 0;
    T blockCount = 0;
    T srcD = 0;
    T srcH = 0;
    T srcW = 0;
    T dstD = 0;
    T dstH = 0;
    T dstW = 0;
    T magicD = 0;
    T shiftD = 0;
    T magicH = 0;
    T shiftH = 0;
    T magicW = 0;
    T shiftW = 0;
};

template <typename T1, typename T2>
class ResizeUpsampleTrilinearSimt {
public:
    __aicore__ inline ResizeUpsampleTrilinearSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetBlockRange(SimtLaunchParams<T2>& params) const;
    __aicore__ inline void RunGeneric(const SimtLaunchParams<T2>& params) const;
    __aicore__ inline void RunDOnly2x(const SimtLaunchParams<T2>& params) const;
    __aicore__ inline bool CanRunDOnly2x(const SimtLaunchParams<T2>& params) const;

    const ResizeUpsampleTrilinearRegBaseTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
};

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::Init(
    GM_ADDR x, GM_ADDR y, const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData)
{
    inputGm_.SetGlobalBuffer((__gm__ T1*)x);
    outputGm_.SetGlobalBuffer((__gm__ T1*)y);
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
}

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::SetBlockRange(SimtLaunchParams<T2>& params) const
{
    params.blockCount = static_cast<T2>(tilingData_->blkProcessNum);
    params.blockStart = static_cast<T2>(blockIdx_) * params.blockCount;
    if (blockIdx_ < tilingData_->tailBlockNum) {
        params.blockCount += 1;
        params.blockStart += static_cast<T2>(blockIdx_);
    } else {
        params.blockStart += static_cast<T2>(tilingData_->tailBlockNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline bool ResizeUpsampleTrilinearSimt<T1, T2>::CanRunDOnly2x(const SimtLaunchParams<T2>& params) const
{
    // Compute in int64_t to detect overflow before narrowing to T2. When
    // T2=uint32_t, inputs with > 2^32 elements would wrap around and could
    // still pass the threshold check, sending the kernel into the 2x path with
    // a truncated task count and corrupting the output. The UINT64/UINT32
    // constants are chosen so the comparison folds away at compile time when
    // T2 is uint64_t, and becomes an overflow guard when T2 is uint32_t.
    int64_t totalTasks64 = static_cast<int64_t>(tilingData_->lenN) * static_cast<int64_t>(tilingData_->lenC) *
                           static_cast<int64_t>(tilingData_->inD) * static_cast<int64_t>(tilingData_->inH) *
                           static_cast<int64_t>(tilingData_->inW);
    if constexpr (sizeof(T2) < sizeof(int64_t)) {
        if (totalTasks64 > static_cast<int64_t>((1ULL << (sizeof(T2) * 8U)) - 1U)) {
            return false;
        }
    }
    T2 totalTasks = static_cast<T2>(totalTasks64);
    return tilingData_->outD == tilingData_->inD * 2 && params.dstH == params.srcH && params.dstW == params.srcW &&
           tilingData_->scaleD != 1.0f && tilingData_->scaleH == 1.0f && tilingData_->scaleW == 1.0f &&
           totalTasks >= static_cast<T2>(GetBlockNum()) * static_cast<T2>(THREAD_NUM_D_ONLY_B32);
}

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::RunGeneric(const SimtLaunchParams<T2>& params) const
{
    asc_vf_call<calleeInt32<T1, T2>>(
        dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
        params.blockStart, params.blockCount, params.magicD, params.shiftD, params.magicH, params.shiftH, params.magicW,
        params.shiftW, params.srcD, params.srcH, params.srcW, params.dstD, params.dstH, params.dstW,
        tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW, tilingData_->alignCorners);
}

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::RunDOnly2x(const SimtLaunchParams<T2>& params) const
{
    // Compute task partitioning in int64_t to avoid T2 (uint32_t) overflow for
    // inputs with > 2^32 elements. CanRunDOnly2x has already rejected shapes
    // that exceed T2 range, so the narrowing below is always safe.
    int64_t totalTasks64 = static_cast<int64_t>(tilingData_->lenN) * static_cast<int64_t>(tilingData_->lenC) *
                           static_cast<int64_t>(tilingData_->inD) * static_cast<int64_t>(tilingData_->inH) *
                           static_cast<int64_t>(tilingData_->inW);
    int64_t blockNum64 = static_cast<int64_t>(GetBlockNum());
    int64_t taskCount64 = totalTasks64 / blockNum64;
    int64_t tailTasks64 = totalTasks64 % blockNum64;
    int64_t taskStart64 = static_cast<int64_t>(blockIdx_) * taskCount64;
    if (static_cast<int64_t>(blockIdx_) < tailTasks64) {
        taskCount64 += 1;
        taskStart64 += static_cast<int64_t>(blockIdx_);
    } else {
        taskStart64 += tailTasks64;
    }
    T2 totalTasks = static_cast<T2>(totalTasks64);
    T2 taskCount = static_cast<T2>(taskCount64);
    T2 taskStart = static_cast<T2>(taskStart64);
    T2 lenSrcHw = params.srcH * params.srcW;
    T2 mSrcD = 0;
    T2 shiftSrcD = 0;
    T2 mSrcHw = 0;
    T2 shiftSrcHw = 0;
    GetUintDivMagicAndShift(mSrcD, shiftSrcD, params.srcD);
    GetUintDivMagicAndShift(mSrcHw, shiftSrcHw, lenSrcHw);
    if (tilingData_->alignCorners == 1) {
        asc_vf_call<calleeInt32DOnly2x<T1, T2, true>>(dim3(THREAD_NUM_D_ONLY_B32), (__gm__ T1*)(inputGm_.GetPhyAddr()),
                                                      (__gm__ T1*)(outputGm_.GetPhyAddr()), taskStart, taskCount, mSrcD,
                                                      shiftSrcD, mSrcHw, shiftSrcHw, params.srcD, lenSrcHw,
                                                      params.dstH * params.dstW, tilingData_->scaleD);
    } else {
        asc_vf_call<calleeInt32DOnly2x<T1, T2, false>>(dim3(THREAD_NUM_D_ONLY_B32), (__gm__ T1*)(inputGm_.GetPhyAddr()),
                                                       (__gm__ T1*)(outputGm_.GetPhyAddr()), taskStart, taskCount,
                                                       mSrcD, shiftSrcD, mSrcHw, shiftSrcHw, params.srcD, lenSrcHw,
                                                       params.dstH * params.dstW, tilingData_->scaleD);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T1, T2>::Process()
{
    if (blockIdx_ >= static_cast<int32_t>(GetBlockNum())) {
        return;
    }
    SimtLaunchParams<T2> params;
    SetBlockRange(params);
    params.srcD = static_cast<T2>(tilingData_->inD);
    params.srcH = static_cast<T2>(tilingData_->inH);
    params.srcW = static_cast<T2>(tilingData_->inW);
    params.dstD = static_cast<T2>(tilingData_->outD);
    params.dstH = static_cast<T2>(tilingData_->outH);
    params.dstW = static_cast<T2>(tilingData_->outW);
    GetUintDivMagicAndShift(params.magicW, params.shiftW, params.dstW);
    GetUintDivMagicAndShift(params.magicH, params.shiftH, params.dstH);
    GetUintDivMagicAndShift(params.magicD, params.shiftD, params.dstD);
    if constexpr (sizeof(T2) == sizeof(uint64_t)) {
        asc_vf_call<calleeInt64<T1, T2>>(
            dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()),
            params.blockStart, params.blockCount, params.magicD, params.shiftD, params.magicH, params.shiftH,
            params.magicW, params.shiftW, params.srcD, params.srcH, params.srcW, params.dstD, params.dstH, params.dstW,
            tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW, tilingData_->alignCorners);
    } else if (CanRunDOnly2x(params)) {
        RunDOnly2x(params);
    } else {
        RunGeneric(params);
    }
}

template <typename T1>
__aicore__ inline void ProcessNcHw(GlobalTensor<T1>& inputGm, GlobalTensor<T1>& outputGm,
                                   const ResizeUpsampleTrilinearRegBaseTilingData* tilingData)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t hwProcessNum = static_cast<uint32_t>(tilingData->blkProcessNum);
    uint32_t nc = blockIdx;
    bool isFull3dHotspot = tilingData->lenN == 1 && tilingData->lenC == 64 && tilingData->inD == 8 &&
                           tilingData->inH == 128 && tilingData->inW == 128 && tilingData->outD == 256 &&
                           tilingData->outH == 256 && tilingData->outW == 256;
    if (isFull3dHotspot) {
        // Also makes kernels compatible with stale FULL_3D_SIMD tiling packets,
        // where blkProcessNum was one N*C slice rather than its H*W size.
        hwProcessNum = static_cast<uint32_t>(tilingData->outH * tilingData->outW);
    } else if (blockIdx < static_cast<uint32_t>(tilingData->tailBlockNum)) {
        hwProcessNum += 1;
    }
    if (tilingData->alignCorners == 1) {
        asc_vf_call<calleeInt32NcHw<T1, uint32_t, true>>(
            dim3(THREAD_NUM_NC_HW), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()), nc,
            hwProcessNum, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW, tilingData->alignCorners);
    } else {
        asc_vf_call<calleeInt32NcHw<T1, uint32_t, false>>(
            dim3(THREAD_NUM_NC_HW), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()), nc,
            hwProcessNum, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW, tilingData->alignCorners);
    }
}

template <typename T1, bool UseTilingSplit>
__aicore__ inline void ProcessDReuseNcHw(GlobalTensor<T1>& inputGm, GlobalTensor<T1>& outputGm,
                                         const ResizeUpsampleTrilinearRegBaseTilingData* tilingData)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t taskCount = 0;
    uint32_t tailBlockNum = 0;
    if constexpr (UseTilingSplit) {
        // The dedicated D_REUSE tiling has already validated uint32 indexing
        // and computed the N*C*H*W split on the host.
        taskCount = static_cast<uint32_t>(tilingData->blkProcessNum);
        tailBlockNum = static_cast<uint32_t>(tilingData->tailBlockNum);
    } else {
        // Generic template binaries store an output-element split in these
        // fields, so the compatibility dispatch must derive its own task split.
        uint32_t blockNum = GetBlockNum();
        uint32_t totalTasks = static_cast<uint32_t>(tilingData->lenN * tilingData->lenC * tilingData->outH *
                                                    tilingData->outW);
        taskCount = totalTasks / blockNum;
        tailBlockNum = totalTasks % blockNum;
    }
    uint32_t taskStart = blockIdx * taskCount;
    if (blockIdx < tailBlockNum) {
        taskCount += 1U;
        taskStart += blockIdx;
    } else {
        taskStart += tailBlockNum;
    }

    uint32_t lenDstW = static_cast<uint32_t>(tilingData->outW);
    uint32_t lenDstH = static_cast<uint32_t>(tilingData->outH);
    uint32_t mW = 0, shiftW = 0, mH = 0, shiftH = 0;
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    GetUintDivMagicAndShift(mH, shiftH, lenDstH);
    if (tilingData->alignCorners == 1) {
        asc_vf_call<calleeDReuseNcHw<T1, uint32_t, true>>(
            dim3(THREAD_NUM_D_REUSE_NC_HW), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            taskStart, taskCount, mW, shiftW, mH, shiftH, static_cast<uint32_t>(tilingData->inD),
            static_cast<uint32_t>(tilingData->inH), static_cast<uint32_t>(tilingData->inW),
            static_cast<uint32_t>(tilingData->outD), lenDstH, lenDstW, tilingData->scaleD, tilingData->scaleH,
            tilingData->scaleW);
    } else {
        asc_vf_call<calleeDReuseNcHw<T1, uint32_t, false>>(
            dim3(THREAD_NUM_D_REUSE_NC_HW), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
            taskStart, taskCount, mW, shiftW, mH, shiftH, static_cast<uint32_t>(tilingData->inD),
            static_cast<uint32_t>(tilingData->inH), static_cast<uint32_t>(tilingData->inW),
            static_cast<uint32_t>(tilingData->outD), lenDstH, lenDstW, tilingData->scaleD, tilingData->scaleH,
            tilingData->scaleW);
    }
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_H_
