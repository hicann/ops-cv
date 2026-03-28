/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_sampler3_d_grad_simt.h
 * \brief
 */
#ifndef GRID_SAMPLER2D_GRAD_SIMT_H_
#define GRID_SAMPLER2D_GRAD_SIMT_H_

#include "simt_api/asc_simt.h"
#include "kernel_operator.h"
#ifdef __CCE_KT_TEST__	 
#include "../../../grid_sampler3_d_grad/op_kernel/arch35/grid_sampler3_d_grad_simt_base.h"
#else 
#include "../../grid_sampler3_d_grad/arch35/grid_sampler3_d_grad_simt_base.h"
#endif

using namespace AscendC;
using namespace GridSampler3DGradSimtBase;
namespace GridSampler2DSimtA5 {
constexpr int32_t GM_PARAMS_SIZE = 5;
constexpr uint32_t VF_MAX_THREAD_NUM = 1024;

template <typename T>
class GridSampler2DGradSimt {
public:
    __aicore__ inline GridSampler2DGradSimt(){};
    __aicore__ inline void Init(
        const GridSampler2DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1]);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGm[GM_PARAMS_SIZE];
    uint32_t blockId_ = GetBlockIdx();
    const GridSampler2DGradTilingData* tiling_;
};

template <typename T>
__aicore__ inline void GridSampler2DGradSimt<T>::Init(
    const GridSampler2DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1])
{
    tiling_ = tilingData;
    // init inputTensor
    inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRAD_INPUT_INDEX]));
    inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[X_INPUT_INDEX]));
    inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRID_INPUT_INDEX]));
    inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DX_INPUT_INDEX]));
    inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DGRID_INPUT_INDEX]));
}

template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputePoints(
        __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, float iy, float ix, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t channelIndex, uint32_t heightCol,
        uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xH, uint32_t xW, uint32_t channel, float tnw, float tne, float tsw, float tse, int32_t iy_tnw, int32_t ix_tnw,
        float* gix, float* giy)
{
    float tnwGradOutValue = static_cast<float>(0.0);
    uint32_t tnwDxIndex = static_cast<uint32_t>(-100);
    GetGradOutValueAndDxIndex(gradOutGmAddr, iy_tnw, ix_tnw, gridH, gridW, batchNum, heightCol, widthCol, channelIndex, newInputIndex, xH, xW, channel, &tnwGradOutValue, &tnwDxIndex);
    if (tnwDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tnwDxIndex, static_cast<T>(tnw * tnwGradOutValue));
        float tnw_val = xGmAddr[tnwDxIndex];
        *gix -= tnw_val * (iy_tnw + 1 - iy) * tnwGradOutValue;
        *giy -= tnw_val * (ix_tnw + 1 - ix) * tnwGradOutValue;
    }

    // tne
    float tneGradOutValue = static_cast<float>(0.0);
    uint32_t tneDxIndex = static_cast<uint32_t>(-100);
    GetGradOutValueAndDxIndex(gradOutGmAddr, iy_tnw, ix_tnw + 1, gridH, gridW, batchNum, heightCol, widthCol, channelIndex, newInputIndex, xH, xW, channel, &tneGradOutValue, &tneDxIndex);
    if (tneDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tneDxIndex, static_cast<T>(tne * tneGradOutValue));
        float tne_val = xGmAddr[tneDxIndex];
        *gix += tne_val * (iy_tnw + 1 - iy) * tneGradOutValue;
        *giy -= tne_val * (ix - ix_tnw) * tneGradOutValue;
    }

    // tsw
    float tswGradOutValue = static_cast<float>(0.0);
    uint32_t tswDxIndex = static_cast<uint32_t>(-100);
    GetGradOutValueAndDxIndex(gradOutGmAddr, iy_tnw + 1, ix_tnw, gridH, gridW, batchNum, heightCol, widthCol, channelIndex, newInputIndex, xH, xW, channel, &tswGradOutValue, &tswDxIndex);
    if (tswDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tswDxIndex, static_cast<T>(tsw * tswGradOutValue));
        float tsw_val = xGmAddr[tswDxIndex];
        *gix -= tsw_val * (iy - iy_tnw) * tswGradOutValue;
        *giy += tsw_val * (ix_tnw + 1 - ix) * tswGradOutValue;
    }

    // tse
    float tseGradOutValue = static_cast<float>(0.0);
    uint32_t tseDxIndex = static_cast<uint32_t>(-100);
    GetGradOutValueAndDxIndex(gradOutGmAddr, iy_tnw + 1, ix_tnw + 1, gridH, gridW, batchNum, heightCol, widthCol, channelIndex, newInputIndex, xH, xW, channel, &tseGradOutValue, &tseDxIndex);
    if (tseDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tseDxIndex, static_cast<T>(tse * tseGradOutValue));
        float tse_val = xGmAddr[tseDxIndex];
        *gix += tse_val * (iy - iy_tnw) * tseGradOutValue;
        *giy += tse_val * (ix - ix_tnw) * tseGradOutValue;
    }
}


template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeBilinear(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, 
	float iy, float ix, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t heightCol, uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, 
	uint32_t xH, uint32_t xW, uint32_t channel, float* ixGradMultValue, float* iyGradMultValue)
{
    int32_t ix_tnw = GetFloorValue(ix);
    int32_t iy_tnw = GetFloorValue(iy);

    // get surfaces to each neighbor:
    float tnw = (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy);
    float tne = (ix - ix_tnw) * (iy_tnw + 1 - iy);
    float tsw = (ix_tnw + 1 - ix) * (iy - iy_tnw);
    float tse = (ix - ix_tnw) * (iy - iy_tnw);

    float gix = static_cast<float>(0);
    float giy = static_cast<float>(0);

    // calculate and set grad_input.
	for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
		ComputePoints(
 	             (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, iy, ix, gridH, gridW,
 	             batchNum, channelIndex, heightCol, widthCol, newInputIndex, offsetBaseAddr, xH, xW, channel,
 	             tnw, tne, tsw, tse, iy_tnw, ix_tnw, &gix, &giy);

		dgridGmAddr[offsetBaseAddr] = static_cast<T>((*ixGradMultValue) * gix);
		dgridGmAddr[offsetBaseAddr + 1] = static_cast<T>((*iyGradMultValue) * giy);
	}
}

template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeNearest(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, float iy, float ix,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xH, uint32_t xW,
    uint32_t channel)
{
    int32_t ix_nearest = Simt::Rint(ix);
    int32_t iy_nearest = Simt::Rint(iy);

    for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
        float gradOutValue = static_cast<float>(0.0);
        uint32_t dxIndex = static_cast<uint32_t>(-100);
        GetGradOutValueAndDxIndex(gradOutGmAddr, iy_nearest, ix_nearest, gridH, gridW, batchNum, heightCol, widthCol, channelIndex, newInputIndex, xH, xW, channel, &gradOutValue, &dxIndex);
        if (dxIndex != -100) {
            Simt::AtomicAdd(dxGmAddr + dxIndex, static_cast<T>(gradOutValue));
        }

        dgridGmAddr[offsetBaseAddr] = static_cast<T>(0);
        dgridGmAddr[offsetBaseAddr + 1] = static_cast<T>(0);
    }
}


// LAUNCH_BOUND
template <typename T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void ComputeGridSampler2DGrad(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* gridGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr,
    uint32_t blockNum, uint32_t batch, uint32_t channel, uint32_t xH, uint32_t xW, uint32_t gridH,
    uint32_t gridW, uint32_t interpolation, uint32_t padding, uint32_t alignCorners, uint32_t gridSize,
    uint32_t shiftH_, uint32_t mH_, uint32_t shiftW_, uint32_t mW_, uint32_t blockId_)
{
    for (uint32_t index = blockId_ * VF_MAX_THREAD_NUM + Simt::GetThreadIdx(); index < gridSize * batch;
         index += (blockNum * VF_MAX_THREAD_NUM)) {
        // output info (N D K_d H K_h W K_w, groups, groupC)
        uint32_t batchNum, heightCol, widthCol;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(index, mH_, shiftH_);
        uint32_t remain = index - batchNum * gridSize;

        heightCol = Simt::UintDiv(remain, mW_, shiftW_);
        widthCol = remain - heightCol * gridW;

        uint32_t newInputIndex = batchNum * channel * xH * xW;
        uint32_t offsetBaseAddr =
            (batchNum * gridH * gridW + heightCol * gridW + widthCol) * 2;

        // get the corresponding input x, y, z co-ordinates from grid
        float ix = static_cast<float>(gridGmAddr[offsetBaseAddr]);     // ix
        float iy = static_cast<float>(gridGmAddr[offsetBaseAddr + 1]); // iy

        // multipliers for gradients on ix, iy
        float ixGradMultValue = 0;
        float iyGradMultValue = 0;

        ix = ComputeSourceIndexSetGrad(ix, xW, padding, alignCorners, &ixGradMultValue);
        iy = ComputeSourceIndexSetGrad(iy, xH, padding, alignCorners, &iyGradMultValue);

        if (interpolation == BILINEAR) {
            // get corner pixel values from (x, y, z)
            // for 5d, we add top-bottom
            ComputeBilinear(
                (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, (__gm__ T*)dgridGmAddr, iy, ix,
                gridH, gridW, batchNum, heightCol, widthCol, newInputIndex, offsetBaseAddr, xH, xW,
                channel, &ixGradMultValue, &iyGradMultValue);
        } else if (interpolation == NEAREST) {
            ComputeNearest(
                (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, (__gm__ T*)dgridGmAddr, iy, ix,
                gridH, gridW, batchNum, heightCol, widthCol, newInputIndex, offsetBaseAddr, xH, xW,
                channel);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DGradSimt<T>::Process()
{
    uint32_t gridSize =  tiling_->gridH * tiling_->gridW;
    uint32_t shiftH_, mH_, shiftW_, mW_;
    GetUintDivMagicAndShift(mH_, shiftH_, static_cast<uint32_t>(tiling_->gridH * tiling_->gridW));
    GetUintDivMagicAndShift(mW_, shiftW_, static_cast<uint32_t>(tiling_->gridW));
    Simt::VF_CALL<ComputeGridSampler2DGrad<T>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ T*)(inputGm[GRAD_INPUT_INDEX].GetPhyAddr()),
        (__gm__ T*)(inputGm[X_INPUT_INDEX].GetPhyAddr()), (__gm__ T*)(inputGm[GRID_INPUT_INDEX].GetPhyAddr()),
        (__gm__ T*)(inputGm[DX_INPUT_INDEX].GetPhyAddr()), (__gm__ T*)(inputGm[DGRID_INPUT_INDEX].GetPhyAddr()),
        tiling_->blockNum, tiling_->batch, tiling_->channel, tiling_->height, tiling_->width, 
        tiling_->gridH, tiling_->gridW, tiling_->interpolation, tiling_->padding, tiling_->alignCorners, gridSize,
        shiftH_, mH_, shiftW_, mW_, blockId_);
}
} // namespace GridSampler2DSimtA5
#endif // GRID_SAMPLER2D_GRAD_SIMT_H_