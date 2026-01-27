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
#ifndef GRID_SAMPLER3D_GRAD_SIMT_H_
#define GRID_SAMPLER3D_GRAD_SIMT_H_

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
constexpr int32_t BILINAER = 0;
constexpr int32_t NEAREST = 1;
constexpr int32_t BORDER = 1;
constexpr int32_t REFLECTION = 2;
constexpr float DEFAULT_FAULT_VALUE = -100.0f;
constexpr uint32_t VF_MAX_THREAD_NUM = 128;

template <typename T>
class GridSampler3DGradSimt {
public:
    __aicore__ inline GridSampler3DGradSimt(){};
    __aicore__ inline void Init(
        const GridSampler3DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1]);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGm[GM_PARAMS_SIZE];
    uint32_t blockId_ = GetBlockIdx();
    const GridSampler3DGradTilingData* tiling_;
};

template <typename T>
__aicore__ inline void GridSampler3DGradSimt<T>::Init(
    const GridSampler3DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1])
{
    tiling_ = tilingData;
    // init inputTensor
    inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRAD_INPUT_INDEX]));
    inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[X_INPUT_INDEX]));
    inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRID_INPUT_INDEX]));
    inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DX_INPUT_INDEX]));
    inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DGRID_INPUT_INDEX]));
}

__aicore__ __attribute__((always_inline)) inline float UnnormallizeSetGrad(
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
    __gm__ T* gradOutGmAddr, int32_t inputXDepth, int32_t inputXHeight, int32_t inputXWidth, uint32_t gridD,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t depthCol, uint32_t heightCol, uint32_t widthCol,
    uint32_t channelIndex, uint32_t gridoffsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW, uint32_t channel)
{
    if (inputXDepth >= 0 && inputXHeight >= 0 && inputXWidth >= 0 && inputXDepth < xD && inputXHeight < xH &&
        inputXWidth < xW) {
        uint32_t gradOutValueIndex = batchNum * channel * gridD * gridH * gridW + channelIndex * gridD * gridH * gridW +
                                     depthCol * gridH * gridW + heightCol * gridW + widthCol;
        return gradOutGmAddr[gradOutValueIndex];
    }
    return static_cast<T>(0.0);
}

__aicore__ __attribute__((always_inline)) inline int64_t GetDxIndex(
    uint32_t newInputIndex, int32_t inputXDepth, int32_t inputXHeight, int32_t inputXWidth, uint32_t channelIndex,
    uint32_t xD, uint32_t xH, uint32_t xW)
{
    if (inputXDepth >= 0 && inputXHeight >= 0 && inputXWidth >= 0 && inputXDepth < xD && inputXHeight < xH &&
        inputXWidth < xW) {
        uint32_t dxIndex =
            newInputIndex + channelIndex * xD * xH * xW + inputXDepth * xH * xW + inputXHeight * xW + inputXWidth;
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
    coord = UnnormallizeSetGrad(coord, size, padding, alignCorners, gradInValue);

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
__aicore__ __attribute__((always_inline)) inline void ComputeTopPoints(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, float iz, float iy, float ix, uint32_t gridD,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t channelIndex, uint32_t depthCol, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW,
    uint32_t channel, float tnw, float tne, float tsw, float tse, int32_t iz_tnw, int32_t iy_tnw, int32_t ix_tnw,
    int32_t iz_tne, int32_t iy_tne, int32_t ix_tne, int32_t iz_tsw, int32_t iy_tsw, int32_t ix_tsw, int32_t iz_tse,
    int32_t iy_tse, int32_t ix_tse, int32_t iz_bnw, int32_t iy_bnw, int32_t ix_bnw, int32_t iz_bne, int32_t iy_bne,
    int32_t ix_bne, int32_t iz_bsw, int32_t iy_bsw, int32_t ix_bsw, int32_t iz_bse, int32_t iy_bse, int32_t ix_bse,
    float* gix, float* giy, float* giz)
{
    // tnw
    float tnwGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_tnw, iy_tnw, ix_tnw, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float tnwDxValue = tnw * tnwGradOutValue;
    int64_t tnwDxIndex = GetDxIndex(newInputIndex, iz_tnw, iy_tnw, ix_tnw, channelIndex, xD, xH, xW);
    if (tnwDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tnwDxIndex, static_cast<T>(tnwDxValue));
        float tnw_val = static_cast<float>(xGmAddr[tnwDxIndex]);
        *gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * tnwGradOutValue;
        *giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * tnwGradOutValue;
        *giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * tnwGradOutValue;
    }

    // tne
    float tneGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_tne, iy_tne, ix_tne, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float tneDxValue = tne * tneGradOutValue;
    int64_t tneDxIndex = GetDxIndex(newInputIndex, iz_tne, iy_tne, ix_tne, channelIndex, xD, xH, xW);
    if (tneDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tneDxIndex, static_cast<T>(tneDxValue));
        float tne_val = static_cast<float>(xGmAddr[tneDxIndex]);
        *gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * tneGradOutValue;
        *giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * tneGradOutValue;
        *giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * tneGradOutValue;
    }

    // tsw
    float tswGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_tsw, iy_tsw, ix_tsw, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float tswDxValue = tsw * tswGradOutValue;
    int64_t tswDxIndex = GetDxIndex(newInputIndex, iz_tsw, iy_tsw, ix_tsw, channelIndex, xD, xH, xW);
    if (tswDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tswDxIndex, static_cast<T>(tswDxValue));
        float tsw_val = static_cast<float>(xGmAddr[tswDxIndex]);
        *gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * tswGradOutValue;
        *giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * tswGradOutValue;
        *giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * tswGradOutValue;
    }

    // tse
    float tseGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_tse, iy_tse, ix_tse, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float tseDxValue = tse * tseGradOutValue;
    int64_t tseDxIndex = GetDxIndex(newInputIndex, iz_tse, iy_tse, ix_tse, channelIndex, xD, xH, xW);
    if (tseDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + tseDxIndex, static_cast<T>(tseDxValue));
        float tse_val = static_cast<float>(xGmAddr[tseDxIndex]);
        *gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * tseGradOutValue;
        *giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * tseGradOutValue;
        *giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * tseGradOutValue;
    }
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void ComputeButtomPoints(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, float iz, float iy, float ix, uint32_t gridD,
    uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t channelIndex, uint32_t depthCol, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW,
    uint32_t channel, float bnw, float bne, float bsw, float bse, int32_t iz_tnw, int32_t iy_tnw, int32_t ix_tnw,
    int32_t iz_tne, int32_t iy_tne, int32_t ix_tne, int32_t iz_tsw, int32_t iy_tsw, int32_t ix_tsw, int32_t iz_tse,
    int32_t iy_tse, int32_t ix_tse, int32_t iz_bnw, int32_t iy_bnw, int32_t ix_bnw, int32_t iz_bne, int32_t iy_bne,
    int32_t ix_bne, int32_t iz_bsw, int32_t iy_bsw, int32_t ix_bsw, int32_t iz_bse, int32_t iy_bse, int32_t ix_bse,
    float* gix, float* giy, float* giz)
{
    // bnw
    float bnwGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_bnw, iy_bnw, ix_bnw, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float bnwDxValue = bnw * bnwGradOutValue;
    int64_t bnwDxIndex = GetDxIndex(newInputIndex, iz_bnw, iy_bnw, ix_bnw, channelIndex, xD, xH, xW);
    if (bnwDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + bnwDxIndex, static_cast<T>(bnwDxValue));
        float bnw_val = static_cast<float>(xGmAddr[bnwDxIndex]);
        *gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * bnwGradOutValue;
        *giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * bnwGradOutValue;
        *giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * bnwGradOutValue;
    }

    // bne
    float bneGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_bne, iy_bne, ix_bne, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float bneDxValue = bne * bneGradOutValue;
    int64_t bneDxIndex = GetDxIndex(newInputIndex, iz_bne, iy_bne, ix_bne, channelIndex, xD, xH, xW);
    if (bneDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + bneDxIndex, static_cast<T>(bneDxValue));
        float bne_val = static_cast<float>(xGmAddr[bneDxIndex]);
        *gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * bneGradOutValue;
        *giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * bneGradOutValue;
        *giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * bneGradOutValue;
    }

    // bsw
    float bswGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_bsw, iy_bsw, ix_bsw, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float bswDxValue = bsw * bswGradOutValue;
    int64_t bswDxIndex = GetDxIndex(newInputIndex, iz_bsw, iy_bsw, ix_bsw, channelIndex, xD, xH, xW);
    if (bswDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + bswDxIndex, static_cast<T>(bswDxValue));
        float bsw_val = static_cast<float>(xGmAddr[bswDxIndex]);
        *gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * bswGradOutValue;
        *giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * bswGradOutValue;
        *giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * bswGradOutValue;
    }

    // bse
    float bseGradOutValue = static_cast<float>(GetGradOutPointValue(
        gradOutGmAddr, iz_bse, iy_bse, ix_bse, gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol,
        channelIndex, offsetBaseAddr, xD, xH, xW, channel));
    float bseDxValue = bse * bseGradOutValue;
    int64_t bseDxIndex = GetDxIndex(newInputIndex, iz_bse, iy_bse, ix_bse, channelIndex, xD, xH, xW);
    if (bseDxIndex != -100) {
        Simt::AtomicAdd(dxGmAddr + bseDxIndex, static_cast<T>(bseDxValue));
        float bse_val = static_cast<float>(xGmAddr[bseDxIndex]);
        *gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * bseGradOutValue;
        *giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * bseGradOutValue;
        *giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * bseGradOutValue;
    }
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void ComputeBilinear(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, float iz, float iy, float ix,
    uint32_t gridD, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t depthCol, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW,
    uint32_t channel, float* ixGradMultValue, float* iyGradMultValue, float* izGradMultValue)
{
    int32_t ix_tnw = GetFloorValue(ix);
    int32_t iy_tnw = GetFloorValue(iy);
    int32_t iz_tnw = GetFloorValue(iz);

    int32_t ix_tne = ix_tnw + 1, iy_tne = iy_tnw, iz_tne = iz_tnw;

    int32_t ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1, iz_tsw = iz_tnw;

    int32_t ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1, iz_tse = iz_tnw;

    int32_t ix_bnw = ix_tnw, iy_bnw = iy_tnw, iz_bnw = iz_tnw + 1;

    int32_t ix_bne = ix_tnw + 1, iy_bne = iy_tnw, iz_bne = iz_tnw + 1;

    int32_t ix_bsw = ix_tnw, iy_bsw = iy_tnw + 1, iz_bsw = iz_tnw + 1;

    int32_t ix_bse = ix_tnw + 1, iy_bse = iy_tnw + 1, iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    float gix = static_cast<float>(0);
    float giy = static_cast<float>(0);
    float giz = static_cast<float>(0);

    // calculate and set grad_input.
    for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
        ComputeTopPoints(
            (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, iz, iy, ix, gridD, gridH, gridW,
            batchNum, channelIndex, depthCol, heightCol, widthCol, newInputIndex, offsetBaseAddr, xD, xH, xW, channel,
            tnw, tne, tsw, tse, iz_tnw, iy_tnw, ix_tnw, iz_tne, iy_tne, ix_tne, iz_tsw, iy_tsw, ix_tsw, iz_tse, iy_tse,
            ix_tse, iz_bnw, iy_bnw, ix_bnw, iz_bne, iy_bne, ix_bne, iz_bsw, iy_bsw, ix_bsw, iz_bse, iy_bse, ix_bse,
            &gix, &giy, &giz);
        ComputeButtomPoints(
            (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, iz, iy, ix, gridD, gridH, gridW,
            batchNum, channelIndex, depthCol, heightCol, widthCol, newInputIndex, offsetBaseAddr, xD, xH, xW, channel,
            bnw, bne, bsw, bse, iz_tnw, iy_tnw, ix_tnw, iz_tne, iy_tne, ix_tne, iz_tsw, iy_tsw, ix_tsw, iz_tse, iy_tse,
            ix_tse, iz_bnw, iy_bnw, ix_bnw, iz_bne, iy_bne, ix_bne, iz_bsw, iy_bsw, ix_bsw, iz_bse, iy_bse, ix_bse,
            &gix, &giy, &giz);

        dgridGmAddr[offsetBaseAddr] = static_cast<T>((*ixGradMultValue) * gix);
        dgridGmAddr[offsetBaseAddr + 1] = static_cast<T>((*iyGradMultValue) * giy);
        dgridGmAddr[offsetBaseAddr + 2] = static_cast<T>((*izGradMultValue) * giz);
    }
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void ComputeNearest(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr, float iz, float iy, float ix,
    uint32_t gridD, uint32_t gridH, uint32_t gridW, uint32_t batchNum, uint32_t depthCol, uint32_t heightCol,
    uint32_t widthCol, uint32_t newInputIndex, uint32_t offsetBaseAddr, uint32_t xD, uint32_t xH, uint32_t xW,
    uint32_t channel)
{
    int32_t ix_nearest = Simt::Rint(ix);
    int32_t iy_nearest = Simt::Rint(iy);
    int32_t iz_nearest = Simt::Rint(iz);

    for (uint32_t channelIndex = 0; channelIndex < channel; channelIndex++) {
        float gradOutValue = static_cast<float>(GetGradOutPointValue(
            gradOutGmAddr, iz_nearest, iy_nearest, ix_nearest, gridD, gridH, gridW, batchNum, depthCol, heightCol,
            widthCol, channelIndex, offsetBaseAddr, xD, xH, xW, channel));
        int64_t dxIndex = GetDxIndex(newInputIndex, iz_nearest, iy_nearest, ix_nearest, channelIndex, xD, xH, xW);
        if (dxIndex != -100) {
            Simt::AtomicAdd(dxGmAddr + dxIndex, static_cast<T>(gradOutValue));
        }

        dgridGmAddr[offsetBaseAddr] = static_cast<T>(0);
        dgridGmAddr[offsetBaseAddr + 1] = static_cast<T>(0);
        dgridGmAddr[offsetBaseAddr + 2] = static_cast<T>(0);
    }
}

// LAUNCH_BOUND
template <typename T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void ComputeGridSampler3DGrad(
    __gm__ T* gradOutGmAddr, __gm__ T* xGmAddr, __gm__ T* gridGmAddr, __gm__ T* dxGmAddr, __gm__ T* dgridGmAddr,
    int64_t blockNum, int64_t batch, int64_t channel, int64_t xD, int64_t xH, int64_t xW, int64_t gridD, int64_t gridH,
    int64_t gridW, int64_t interpolation, int64_t padding, int64_t alignCorners, uint32_t gridSize, uint32_t shiftD_,
    uint32_t mD_, uint32_t shiftH_, uint32_t mH_, uint32_t shiftW_, uint32_t mW_, uint32_t blockId_)
{
    for (uint32_t index = blockId_ * VF_MAX_THREAD_NUM + Simt::GetThreadIdx(); index < gridSize * batch;
         index += (blockNum * VF_MAX_THREAD_NUM)) {
        // output info (N D K_d H K_h W K_w, groups, groupC)
        uint32_t batchNum, depthCol, heightCol, widthCol;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(index, mD_, shiftD_);
        uint32_t remain = index - batchNum * gridSize;

        depthCol = Simt::UintDiv(remain, mH_, shiftH_);
        remain = remain - depthCol * (gridH * gridW);

        heightCol = Simt::UintDiv(remain, mW_, shiftW_);
        widthCol = remain - heightCol * gridW;

        uint32_t newInputIndex = batchNum * channel * xD * xH * xW;
        uint32_t offsetBaseAddr =
            (batchNum * gridD * gridH * gridW + depthCol * gridH * gridW + heightCol * gridW + widthCol) * 3;

        // get the corresponding input x, y, z co-ordinates from grid
        float ix = static_cast<float>(gridGmAddr[offsetBaseAddr]);     // ix
        float iy = static_cast<float>(gridGmAddr[offsetBaseAddr + 1]); // iy
        float iz = static_cast<float>(gridGmAddr[offsetBaseAddr + 2]); // iz

        // multipliers for gradients on ix, iy, and iz
        float ixGradMultValue = 0;
        float iyGradMultValue = 0;
        float izGradMultValue = 0;

        ix = ComputeSourceIndexSetGrad(ix, xW, padding, alignCorners, &ixGradMultValue);
        iy = ComputeSourceIndexSetGrad(iy, xH, padding, alignCorners, &iyGradMultValue);
        iz = ComputeSourceIndexSetGrad(iz, xD, padding, alignCorners, &izGradMultValue);

        if (interpolation == BILINAER) {
            // get corner pixel values from (x, y, z)
            // for 5d, we add top-bottom
            ComputeBilinear(
                (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, (__gm__ T*)dgridGmAddr, iz, iy, ix,
                gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol, newInputIndex, offsetBaseAddr, xD, xH, xW,
                channel, &ixGradMultValue, &iyGradMultValue, &izGradMultValue);
        } else if (interpolation == NEAREST) {
            ComputeNearest(
                (__gm__ T*)gradOutGmAddr, (__gm__ T*)xGmAddr, (__gm__ T*)dxGmAddr, (__gm__ T*)dgridGmAddr, iz, iy, ix,
                gridD, gridH, gridW, batchNum, depthCol, heightCol, widthCol, newInputIndex, offsetBaseAddr, xD, xH, xW,
                channel);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradSimt<T>::Process()
{
    uint32_t gridSize = tiling_->gridD * tiling_->gridH * tiling_->gridW;
    uint32_t shiftD_, mD_, shiftH_, mH_, shiftW_, mW_;
    GetUintDivMagicAndShift(mD_, shiftD_, gridSize);
    GetUintDivMagicAndShift(mH_, shiftH_, static_cast<uint32_t>(tiling_->gridH * tiling_->gridW));
    GetUintDivMagicAndShift(mW_, shiftW_, static_cast<uint32_t>(tiling_->gridW));
    Simt::VF_CALL<ComputeGridSampler3DGrad<T>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ T*)(inputGm[GRAD_INPUT_INDEX].GetPhyAddr()),
        (__gm__ T*)(inputGm[X_INPUT_INDEX].GetPhyAddr()), (__gm__ T*)(inputGm[GRID_INPUT_INDEX].GetPhyAddr()),
        (__gm__ T*)(inputGm[DX_INPUT_INDEX].GetPhyAddr()), (__gm__ T*)(inputGm[DGRID_INPUT_INDEX].GetPhyAddr()),
        tiling_->blockNum, tiling_->batch, tiling_->channel, tiling_->xD, tiling_->xH, tiling_->xW, tiling_->gridD,
        tiling_->gridH, tiling_->gridW, tiling_->interpolation, tiling_->padding, tiling_->alignCorners, gridSize,
        shiftD_, mD_, shiftH_, mH_, shiftW_, mW_, blockId_);
}
#endif // GRID_SAMPLER3D_GRAD_SIMT_H_