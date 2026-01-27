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

using namespace AscendC;

constexpr int32_t INT_MAX = 2147483647;
constexpr int32_t INT_MIN = -2147483648;
constexpr int32_t INPUT_NUM = 3;
constexpr int32_t OUTPUT_NUM = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DX_INPUT_INDEX = 3;
constexpr int32_t DGRID_INPUT_INDEX = 4;
constexpr int32_t GM_PARAMS_SIZE = 5;
constexpr int32_t DX_OUTPUT_INDEX = 0;
constexpr int32_t DGRID_OUTPUT_INDEX = 1;
constexpr int32_t BILINEAR = 0;
constexpr int32_t NEAREST = 1;
constexpr int32_t BORDER = 1;
constexpr int32_t REFLECTION = 2;
constexpr float DEFAULT_FAULT_VALUE = -100.0f;
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

__aicore__ __attribute__((always_inline)) inline float UnnormalizeSetGrad(
    float coord, int64_t size, int64_t padding, int64_t alignCorners, float* gradInValue)
{
    if (alignCorners == 1) {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        coord = (coord + 1) / 2 * (size - 1);
        *gradInValue = static_cast<float>(size - 1) / 2;
    } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        coord = ((coord + 1) * size - 1) / 2;
        *gradInValue = static_cast<float>(size) / 2;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float ClipCoorDinatesSetGrad(
    float coord, int64_t clip_limit, float* gradClipValue)
{
    if (coord <= static_cast<float>(0)) {
        *gradClipValue = static_cast<float>(0);
        return static_cast<float>(0);
    } else {
        float max = static_cast<float>(clip_limit - 1);
        if (coord >= max) {
            *gradClipValue = static_cast<float>(0);
            return max;
        } else {
            *gradClipValue = static_cast<float>(1);
            return coord;
        }
    }
}

__aicore__ __attribute__((always_inline)) inline float ReflectCoordinatesSetGrad(
    float coord, int twiceLow, int64_t twiceHigh, float* gradReflValue)
{
    if (twiceLow == twiceHigh) {
        *gradReflValue = static_cast<float>(0);
        return static_cast<float>(0);
    }
    float gradInMult = 0;
    float min = static_cast<float>(twiceLow) / 2;
    float span = static_cast<float>(twiceHigh - twiceLow) / 2;
    coord = coord - min;
    if (coord < static_cast<float>(0)) {
        gradInMult = static_cast<float>(-1);
        coord = -coord;
    } else {
        gradInMult = static_cast<float>(1);
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    float extra = Simt::Mod(coord, span);
    int32_t flips = static_cast<int32_t>(Simt::Floor(coord / span));
    if (flips % 2 == 0) {
        *gradReflValue = gradInMult;
        return extra + min;
    } else {
        *gradReflValue = -gradInMult;
        return span - extra + min;
    }
}

__aicore__ __attribute__((always_inline)) inline float SafeDowngradeToIntRange(float coord)
{
    if (!Simt::IsFinite(coord)) {
        return DEFAULT_FAULT_VALUE;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline int32_t GetFloorValue(float x)
{
    float negativeValue = static_cast<float>(0.0);
    float floorFactor = static_cast<float>(-1);
    return (x >= negativeValue ? static_cast<int32_t>(x) : static_cast<int32_t>(floorFactor + x));
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline T GetGradOutPointValue(
    __gm__ T* gradOutGmAddr, int32_t inputXHeight, int32_t inputXWidth,
    uint32_t gridH, uint32_t gridW, 
	uint32_t batchNum, uint32_t heightCol, uint32_t widthCol, uint32_t channelIndex, uint32_t xH, uint32_t xW, uint32_t channel)
{
    if (inputXHeight >= 0 && inputXWidth >= 0 && inputXHeight < xH && inputXWidth < xW) {
        uint32_t gradOutValueIndex = batchNum * channel * gridH * gridW + channelIndex * gridH * gridW +
                                    heightCol * gridW + widthCol;
        return gradOutGmAddr[gradOutValueIndex];
    }
    return static_cast<T>(0.0);
}

__aicore__ __attribute__((always_inline)) inline int64_t GetDxIndex(
    uint32_t newInputIndex, int32_t inputXHeight, int32_t inputXWidth, uint32_t channelIndex,
	uint32_t xH, uint32_t xW)
{
    if (inputXHeight >= 0 && inputXWidth >= 0 && inputXHeight < xH &&
        inputXWidth < xW) {
        uint32_t dxIndex =
            newInputIndex + channelIndex * xH * xW + inputXHeight * xW + inputXWidth;
        return static_cast<int64_t>(dxIndex);
    } else {
        return -100;
    }
}

__aicore__ __attribute__((always_inline)) inline float ComputeSourceIndexSetGrad(
    float coord, int64_t size, int64_t padding, int64_t alignCorners, float* gradInValue)
{
    float gradClipValue = 0;
    float gradReflValue = 0;
    coord = UnnormalizeSetGrad(coord, size, padding, alignCorners, gradInValue);

    if (padding == BORDER) {
        // clip coordinates to image borders
        coord = ClipCoorDinatesSetGrad(coord, size, &gradClipValue);
        *gradInValue = (*gradInValue) * gradClipValue;
    } else if (padding == REFLECTION) {
        // reflect coordinates by image borders
        if (alignCorners) {
            coord = ReflectCoordinatesSetGrad(coord, 0, 2 * (size - 1), &gradReflValue);
        } else {
            coord = ReflectCoordinatesSetGrad(coord, -1, 2 * size - 1, &gradReflValue);
        }
        // clip coordinates to image borders
        coord = ClipCoorDinatesSetGrad(coord, size, &gradClipValue);
        *gradInValue = (*gradInValue) * gradReflValue * gradClipValue;
    }
    coord = SafeDowngradeToIntRange(coord);
    return coord;
}

template <typename T>
    __aicore__ __attribute__((always_inline)) inline void ComputePoints(
        __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, float iy, float ix, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t channelIndex, uint32_t heightCol,
        uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xH, uint32_t xW, uint32_t channel, float tnw, float tne, float tsw, float tse, int32_t iy_tnw, int32_t ix_tnw,
        int32_t iy_tne, int32_t ix_tne, int32_t iy_tsw, int32_t ix_tsw, int32_t iy_tse, int32_t ix_tse, float* gix, float* giy)
{
    float tnwGradOutValue = static_cast<float>(GetGradOutPointValue(
    gradOutGmAddr, iy_tnw, ix_tnw, gridH, gridW, batchNum, heightCol, widthCol,
    channelIndex, xH, xW, channel));
    float tnwDxValue = tnw * tnwGradOutValue;
    int64_t tnwDxIndex = GetDxIndex(newInputIndex, iy_tnw, ix_tnw, channelIndex, xH, xW);
    if (tnwDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tnwDxIndex, static_cast<T>(tnwDxValue));
        float tnw_val = xGmAddr[tnwDxIndex];
        *gix -= tnw_val * (iy_tse - iy) * tnwGradOutValue;
        *giy -= tnw_val * (ix_tse - ix) * tnwGradOutValue;
    }

    // tne
    float tneGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iy_tne, ix_tne, gridH, gridW, batchNum, heightCol, widthCol,
        channelIndex, xH, xW, channel));
    float tneDxValue = tne * tneGradOutValue;
    int64_t tneDxIndex = GetDxIndex(newInputIndex, iy_tne, ix_tne, channelIndex, xH, xW);
    if (tneDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tneDxIndex, static_cast<T>(tneDxValue));
        float tne_val = xGmAddr[tneDxIndex];
        *gix += tne_val * (iy_tsw - iy) * tneGradOutValue;
        *giy -= tne_val * (ix - ix_tsw) * tneGradOutValue;
    }

    // tsw
    float tswGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iy_tsw, ix_tsw, gridH, gridW, batchNum, heightCol, widthCol,
        channelIndex, xH, xW, channel));
    float tswDxValue = tsw * tswGradOutValue;
    int64_t tswDxIndex = GetDxIndex(newInputIndex, iy_tsw, ix_tsw, channelIndex, xH, xW);
    if (tswDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tswDxIndex, static_cast<T>(tswDxValue));
        float tsw_val = xGmAddr[tswDxIndex];
        *gix -= tsw_val * (iy - iy_tne) * tswGradOutValue;
        *giy += tsw_val * (ix_tne - ix) * tswGradOutValue;
    }

    // tse
    float tseGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iy_tse, ix_tse, gridH, gridW, batchNum, heightCol, widthCol,
        channelIndex, xH, xW, channel));
    float tseDxValue = tse * tseGradOutValue;
    int64_t tseDxIndex = GetDxIndex(newInputIndex, iy_tse, ix_tse, channelIndex, xH, xW);
    if (tseDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tseDxIndex, static_cast<T>(tseDxValue));
        float tse_val = xGmAddr[tseDxIndex];
        *gix += tse_val * (iy - iy_tnw) * tseGradOutValue;
        *giy += tse_val * (ix - ix_tnw) * tseGradOutValue;
    }
}


template <typename T>
__aicore__ __attribute__((always_inline)) inline void ComputeBilinear(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, 
	float iy, float ix, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t heightCol, uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, 
	uint32_t xH, uint32_t xW, uint32_t channel, float* ixGradMultValue, float* iyGradMultValue)
{
    int32_t ix_tnw = GetFloorValue(ix);
    int32_t iy_tnw = GetFloorValue(iy);

    int32_t ix_tne = ix_tnw + 1;
    int32_t iy_tne = iy_tnw;

    int32_t ix_tsw = ix_tnw;
    int32_t iy_tsw = iy_tnw + 1;

    int32_t ix_tse = ix_tnw + 1;
    int32_t iy_tse = iy_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_tse - ix) * (iy_tse - iy);
    float tne = (ix - ix_tsw) * (iy_tsw - iy);
    float tsw = (ix_tne - ix) * (iy - iy_tne);
    float tse = (ix - ix_tnw) * (iy - iy_tnw);

    float gix = static_cast<float>(0);
    float giy = static_cast<float>(0);

    // calculate and set grad_input.
	for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
		ComputePoints(
 	             (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, iy, ix, gridH, gridW,
 	             batchNum, channelIndex, heightCol, widthCol, newInputIndex, offsetBaseAddr, xH, xW, channel,
 	             tnw, tne, tsw, tse, iy_tnw, ix_tnw, iy_tne, ix_tne, iy_tsw, ix_tsw, iy_tse,
 	             ix_tse, &gix, &giy);

		dgridGmAddr[offsetBaseAddr] = static_cast<T>((*ixGradMultValue) * gix);
		dgridGmAddr[offsetBaseAddr + 1] = static_cast<T>((*iyGradMultValue) * giy);
	}
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void ComputeNearest(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, float iy, float ix,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xH, uint32_t xW,
    uint32_t channel)
{
    int32_t ix_nearest = Simt::Rint(ix);
    int32_t iy_nearest = Simt::Rint(iy);

    for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
        float gradOutValue = static_cast<float>(GetGradOutPointValue(
            gradOutGmAddr, iy_nearest, ix_nearest, gridH, gridW, batchNum, heightCol,
            widthCol, channelIndex, xH, xW, channel));
        int64_t dxIndex = GetDxIndex(newInputIndex, iy_nearest, ix_nearest, channelIndex, xH, xW);
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
    int64_t blockNum, int64_t batch, int64_t channel, int64_t xH, int64_t xW, int64_t gridH,
    int64_t gridW, int64_t interpolation, int64_t padding, int64_t alignCorners, uint32_t gridSize,
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
#endif // GRID_SAMPLER2D_GRAD_SIMT_H_