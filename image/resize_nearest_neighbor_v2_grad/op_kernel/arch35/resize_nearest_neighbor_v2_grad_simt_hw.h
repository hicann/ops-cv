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
 * \file resize_nearest_neighbor_v2_grad_simt_hw.h
 * \brief resize_nearest_neighbor_v2_grad_simt_hw
 */

#ifndef CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_HW
#define CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_HW

#include "resize_nearest_neighbor_v2_grad_simt_base.h"

namespace ResizeNearestNeighborV2Grad{
using namespace AscendC;

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
class ResizeNearestNeighborV2GradSimtHW
{
public:
    __aicore__ inline ResizeNearestNeighborV2GradSimtHW(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR y, const ResizeNearestNeighborV2GradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T_DATA> gradsGm_;
    GlobalTensor<T_DATA> yGm_;
    int32_t blockIdx_;
    const ResizeNearestNeighborV2GradTilingData* tilingData_;
};

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__aicore__ inline void ResizeNearestNeighborV2GradSimtHW<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>::Init(
    GM_ADDR grads, GM_ADDR y, const ResizeNearestNeighborV2GradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    gradsGm_.SetGlobalBuffer((__gm__ T_DATA*)grads);
    yGm_.SetGlobalBuffer((__gm__ T_DATA*)y);
}

template <typename T_IDX, bool ALIGN_CORNERS, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline T_IDX CalcSourceIndexHW(float scale, T_IDX dstIdx, T_IDX srcIdxMax)
{
    if constexpr (ALIGN_CORNERS && !HALF_PIXEL) {
        return Simt::Min(static_cast<T_IDX>(Simt::Round(static_cast<float>(dstIdx) * scale)), srcIdxMax);
    } else if constexpr (!ALIGN_CORNERS && HALF_PIXEL) {
        return Simt::Min(static_cast<T_IDX>(Simt::Floor(static_cast<float>(dstIdx + HALF_PIXEL_VAL) * scale)),
                         srcIdxMax);
    } else {
        return Simt::Min(static_cast<T_IDX>(Simt::Floor(static_cast<float>(dstIdx) * scale)), srcIdxMax);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtNCHWCompute(__gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC,
                                                                  T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH,
                                                                  T_IDX lenDstW, float scaleH, float scaleW,
                                                                  T_IDX coreFactor, T_IDX coreOffset, T_IDX mW,
                                                                  T_IDX shiftW)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor; idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) 
    {
        T_IDX gradsIdx = coreOffset + idx;
        T_IDX  gradsIdxH = Simt::UintDiv(gradsIdx, mW, shiftW) ;
        T_IDX  gradsIdxW = gradsIdx - gradsIdxH * lenDstW ; 
        T_IDX lenNC = lenN * lenC;
        for(T_IDX idxNC = 0 ; idxNC < lenNC ; idxNC++){
            T_IDX yIdxH = CalcSourceIndexHW<T_IDX, ALIGN_CORNERS, HALF_PIXEL>(scaleH, gradsIdxH, lenSrcH - 1);
            T_IDX yIdxW = CalcSourceIndexHW<T_IDX, ALIGN_CORNERS, HALF_PIXEL>(scaleW, gradsIdxW, lenSrcW - 1);
            T_IDX yIdx = idxNC * lenSrcH * lenSrcW + yIdxH * lenSrcW + yIdxW;
            Simt::AtomicAdd(y + yIdx, grads[idxNC * lenDstH * lenDstW + gradsIdx]);
        }
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__aicore__ __attribute__((always_inline)) inline void SimtNHWCCompute(__gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC,
                                                                  T_IDX lenSrcH, T_IDX lenSrcW, T_IDX lenDstH,
                                                                  T_IDX lenDstW, float scaleH, float scaleW,
                                                                  T_IDX coreFactor, T_IDX coreOffset, T_IDX mW,
                                                                  T_IDX shiftW , T_IDX mC, T_IDX shiftC)
{
    for (T_IDX idx = static_cast<T_IDX>(Simt::GetThreadIdx()); idx < coreFactor; idx += static_cast<T_IDX>(Simt::GetThreadNum<0>())) 
    {
        T_IDX gradsIdx = coreOffset + idx;
        T_IDX  gradsIdxH = Simt::UintDiv(gradsIdx, mW, shiftW);
        T_IDX  gradsIdxW = gradsIdx - lenDstW * gradsIdxH;       
        for(T_IDX idxN = 0 ; idxN < lenN ; idxN++){
            for(T_IDX idxC = 0 ; idxC < lenC ; idxC++){
                T_IDX yIdxH = CalcSourceIndexHW<T_IDX, ALIGN_CORNERS, HALF_PIXEL>(scaleH, gradsIdxH, lenSrcH - 1);
                T_IDX yIdxW = CalcSourceIndexHW<T_IDX, ALIGN_CORNERS, HALF_PIXEL>(scaleW, gradsIdxW, lenSrcW - 1);
                T_IDX yIdx = ((idxN * lenSrcH + yIdxH) * lenSrcW + yIdxW) * lenC + idxC;
                Simt::AtomicAdd(y + yIdx, grads[idxN * lenDstH * lenDstW * lenC + gradsIdx * lenC +idxC]);
            }
            }
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT32) __aicore__
    void calleeSimtHWInt32(__gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN,  T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW,
                         T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX coreFactor, T_IDX coreOffset,
                         T_IDX mW, T_IDX shiftW, T_IDX mC, T_IDX shiftC)
{
    if constexpr (FORMAT == FORMAT_NCHW){
        SimtNCHWCompute<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>(
            grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, coreFactor, coreOffset, mW, shiftW);
    }else {
        SimtNHWCCompute<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>(
            grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, coreFactor, coreOffset, mW, shiftW, mC,  shiftC);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM_INT64) __aicore__
    void calleeSimtHWInt64(__gm__ T_DATA* grads, __gm__ T_DATA* y, T_IDX lenN, T_IDX lenC, T_IDX lenSrcH, T_IDX lenSrcW,
                         T_IDX lenDstH, T_IDX lenDstW, float scaleH, float scaleW, T_IDX coreFactor, T_IDX coreOffset,
                          T_IDX mW, T_IDX shiftW, T_IDX mC, T_IDX shiftC)
{
    if constexpr (FORMAT == FORMAT_NCHW){
        SimtNCHWCompute<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>(
            grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, coreFactor, coreOffset, mW, shiftW);
    }else {
        SimtNHWCCompute<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>(
            grads, y, lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW, coreFactor, coreOffset, mW, shiftW, mC,  shiftC);
    }
}

template <typename T_DATA, typename T_IDX, int32_t FORMAT, bool ALIGN_CORNERS, bool HALF_PIXEL>
__aicore__ inline void ResizeNearestNeighborV2GradSimtHW<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>::Process()
{
    const T_IDX lenC = tilingData_->lenC;
    const T_IDX lenN = tilingData_->lenN;
    const T_IDX lenSrcW = tilingData_->lenSrcW;
    const T_IDX lenSrcH = tilingData_->lenSrcH;
    const T_IDX lenDstW = tilingData_->lenDstW;
    const T_IDX lenDstH = tilingData_->lenDstH;
    const int32_t initYUseCoreNum = tilingData_->initYRealCoreNum;
    const int32_t useCoreNum = tilingData_->realCoreNum;
    const float scaleH = tilingData_->scaleH;
    const float scaleW = tilingData_->scaleW;

    T_IDX blkProcessNum = tilingData_->splitBlockFactor;
    T_IDX blkStartOffset = blockIdx_ * tilingData_->splitBlockFactor;
    T_IDX blkProcessNumY = tilingData_->initYSplitBlockFactor;
    T_IDX blkStartOffsetY = blockIdx_ * tilingData_->initYSplitBlockFactor;
    if (blockIdx_ < tilingData_->splitBlockTailFactor) {
        blkProcessNum += 1;
        blkStartOffset += blockIdx_;
    } else {
        blkStartOffset += tilingData_->splitBlockTailFactor;
    }
    if (blockIdx_ < tilingData_->initYSplitBlockTailFactor) {
        blkProcessNumY += 1;
        blkStartOffsetY += blockIdx_;
    } else {
        blkStartOffsetY += tilingData_->initYSplitBlockTailFactor;
    }

    T_IDX mW = 1;
    T_IDX mC = 1;
    T_IDX shiftW = 1;
    T_IDX shiftC = 1;
    GetUintDivMagicAndShift(mW, shiftW, lenDstW);
    GetUintDivMagicAndShift(mC, shiftC, lenC);

    if (blockIdx_ < initYUseCoreNum) {
        InitOutput<T_DATA>(yGm_[blkStartOffsetY], blkProcessNumY, static_cast<T_DATA>(0.0f));
    }
    SyncAll();

    if (blockIdx_ < useCoreNum) {
        if constexpr (sizeof(T_IDX) == sizeof(uint32_t)) {
            Simt::VF_CALL<calleeSimtHWInt32<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT32), (__gm__ T_DATA*)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(yGm_.GetPhyAddr()),lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW,
                blkProcessNum, blkStartOffset, mW, shiftW , mC , shiftC);
        } else {
            Simt::VF_CALL<calleeSimtHWInt64<T_DATA, T_IDX, FORMAT, ALIGN_CORNERS, HALF_PIXEL>>(
                Simt::Dim3(SIMT_THREAD_NUM_INT64), (__gm__ T_DATA*)(gradsGm_.GetPhyAddr()),
                (__gm__ T_DATA*)(yGm_.GetPhyAddr()),lenN, lenC, lenSrcH, lenSrcW, lenDstH, lenDstW, scaleH, scaleW,
                blkProcessNum, blkStartOffset, mW, shiftW, mC , shiftC);
        }
    }
}
}// namespace ResizeNearestNeighborV2GradSimtHW

#endif  // CANN_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_SIMT_HW