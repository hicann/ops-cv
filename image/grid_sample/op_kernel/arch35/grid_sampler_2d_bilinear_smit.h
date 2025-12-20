/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grid_sampler_2d.h
 * \brief grid sampler 2d kernel info
 */
#ifndef GRID_SAMPLER_2D_H
#define GRID_SAMPLER_2D_H
#include "kernel_operator.h"
namespace GridSample {

using namespace AscendC;

const uint32_t WIDTH_OFFSET_INDEX = 0;
const uint32_t HEIGHT_OFFSET_INDEX = 1;
const uint32_t POINT_WEIGHT_OFFSET_INDEX = 2;
const uint32_t VF_MAX_THREAD_NUM = 512;
const uint32_t OFFSET_DIM_VALUE = 3;

const uint32_t REFLECT_RATIO_95 = 2;
const uint32_t PADDING_MODE_ZEROS_95 = 0;
const uint32_t PADDING_MODE_BORDER_95 = 1;
const uint32_t PADDING_MODE_REFLECTION_95 = 2;
const uint32_t INT_MAX = 2147483647;
const uint32_t INT_MIN = -2147483648;
const float DEFAULT_FAULT_VALUE = -100.0f;

template <typename T>
class GridSampler2dBilinearSimt {
public:
    __aicore__ inline GridSampler2dBilinearSimt()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace,
        const GridSampler2dTilingDataSimt* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputImgGm_;
    GlobalTensor<T> gridGm_;
    GlobalTensor<T> yGm_;
    uint32_t blockId_ = GetBlockIdx();
    const GridSampler2dTilingDataSimt* tiling_;
};

template <typename T>
__aicore__ inline void GridSampler2dBilinearSimt<T>::Init(
    GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace, const GridSampler2dTilingDataSimt* __restrict tilingData)
{
    inputImgGm_.SetGlobalBuffer((__gm__ T*)(x));
    gridGm_.SetGlobalBuffer((__gm__ T*)(grid));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    tiling_ = tilingData;
}

__aicore__ __attribute__((always_inline)) inline int32_t GetFloorValue(float x)
{
    float negativeValue = static_cast<float>(0.0);
    float floorFactor = static_cast<float>(-1);
    return (x >= negativeValue ? static_cast<int32_t>(x) : static_cast<int32_t>(floorFactor + x));
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline T GetInputPointValue(
    __gm__ T* inputImgGmAddr, int32_t inputHeight, int32_t inputWidth, uint32_t channelIndex,
    uint32_t inputDataBatchOffset, uint32_t inH, uint32_t inW, uint32_t inC)
{
    if (inputHeight >= 0 && inputWidth >= 0 && inputHeight < inH && inputWidth < inW) {
        return inputImgGmAddr[inputDataBatchOffset + inputHeight * inW + inputWidth + channelIndex * inH * inW];
    }
    return static_cast<T>(0.0);
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline T ComputeBilinear(
    __gm__ T* inputImgGmAddr, float pointHeight, float pointWidth, uint32_t channelIndex, uint32_t inputDataBatchOffset,
    uint32_t inH, uint32_t inW, uint32_t inC)
{
    int32_t heightFloor = GetFloorValue(pointHeight);
    int32_t widthFloor = GetFloorValue(pointWidth);

    float heightFloorDelta = pointHeight - static_cast<float>(heightFloor);
    float widthFloorDelta = pointWidth - static_cast<float>(widthFloor);

    // pointLeftUp
    float inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, heightFloor, widthFloor, channelIndex, inputDataBatchOffset, inH, inW, inC));
    float inputWeight = (1.0f - heightFloorDelta) * (1.0f - widthFloorDelta);
    float bilinearValue = (inputValue * inputWeight);

    // pointRightUp
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, heightFloor, (widthFloor + 1), channelIndex, inputDataBatchOffset, inH, inW, inC));
    inputWeight = (1.0f - heightFloorDelta) * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointLeftBottom
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), widthFloor, channelIndex, inputDataBatchOffset, inH, inW, inC));
    inputWeight = heightFloorDelta * (1.0f - widthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointRightBottom
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), (widthFloor + 1), channelIndex, inputDataBatchOffset, inH, inW,
        inC));
    inputWeight = heightFloorDelta * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    return static_cast<T>(bilinearValue);
}

__aicore__ __attribute__((always_inline)) inline float ClipCoordinates(float coord, int64_t size)
{
    coord = coord > (size - 1) ? (float)size - 1 : coord;
    coord = coord < 0 ? 0 : coord;
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float reflectCoordinates(float coord, int twiceLow, int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        return 0;
    }
    float min = static_cast<float>(twiceLow) / 2;
    float span = static_cast<float>(twiceHigh - twiceLow) / 2;
    coord = Simt::Abs(coord - min);
    float extra = Simt::Mod(coord, span);
    int32_t flips = static_cast<int32_t>(Simt::Floor(coord / span));
    if (flips % REFLECT_RATIO_95 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float safeDowngradeToIntRange(float coord)
{
    if (!Simt::IsFinite(coord)) {
        return DEFAULT_FAULT_VALUE;
    }
    return coord;
}

__aicore__ __attribute__((always_inline)) inline float Clip(
    float coord, int64_t size, int64_t paddingMode, int64_t alignCorners)
{
    if (paddingMode == PADDING_MODE_BORDER_95) {
        coord = ClipCoordinates(coord, size);
    } else if (paddingMode == PADDING_MODE_REFLECTION_95) {
        if (alignCorners) {
            coord = reflectCoordinates(coord, 0, REFLECT_RATIO_95 * (size - 1));
        } else {
            coord = reflectCoordinates(coord, -1, REFLECT_RATIO_95 * size - 1);
        }
        coord = ClipCoordinates(coord, size);
    }
    safeDowngradeToIntRange(coord);
    return coord;
}

// LAUNCH_BOUND
template <typename T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void ComputeGridSampler2d(
    __gm__ T* inputImgGmAddr, __gm__ T* gridGmAddr, __gm__ T* yGmAddr, int64_t blockNum, int64_t intN, int64_t inC,
    int64_t inH, int64_t inW, int64_t outH, int64_t outW, int64_t paddingMode, int64_t alignCorners,
    uint32_t outImgSize, uint32_t shiftB_, uint32_t mB_, uint32_t shiftH_, uint32_t mH_, uint32_t shiftW_, uint32_t mW_,
    uint32_t blockId_)
{
    for (uint32_t index = blockId_ * VF_MAX_THREAD_NUM + Simt::GetThreadIdx(); index < outImgSize * intN;
         index += (blockNum * VF_MAX_THREAD_NUM)) {
        // output info (N H K_h W K_w, groups, groupC)
        uint32_t batchNum, heightCol, widthCol, channelIndex;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(index, mB_, shiftB_); // outImgSize tiling_->outW * tiling_->outH * tiling_->inC
        uint32_t remain = index - batchNum * outImgSize;

        channelIndex = Simt::UintDiv(remain, mH_, shiftH_); // tiling_->outW * tiling_->inC
        remain = remain - channelIndex * (outW * outH);

        heightCol = Simt::UintDiv(remain, mW_, shiftW_); // tiling_->inC
        widthCol = remain - heightCol * outW;

        uint32_t newInputIndex = batchNum * inH * inW * inC;
        uint32_t offsetBaseAdrr = batchNum * outH * outW * 2 + heightCol * outW * 2 + widthCol * 2;
        // offset height info
        float gridHeigthValue = static_cast<float>(gridGmAddr[offsetBaseAdrr + 1]);
        float gridWeightValue = static_cast<float>(gridGmAddr[offsetBaseAdrr]);
        if (alignCorners == 1) {
            gridHeigthValue = (gridHeigthValue + 1) / 2 * (inH - 1);
            gridWeightValue = (gridWeightValue + 1) / 2 * (inW - 1);
        } else {
            gridHeigthValue = ((gridHeigthValue + 1) * inH - 1) / 2;
            gridWeightValue = ((gridWeightValue + 1) * inW - 1) / 2;
        }

        gridWeightValue = Clip(gridWeightValue, inW, paddingMode, alignCorners);
        gridHeigthValue = Clip(gridHeigthValue, inH, paddingMode, alignCorners);

        T bilinearValue = ComputeBilinear(
            (__gm__ T*)(inputImgGmAddr), gridHeigthValue, gridWeightValue, channelIndex, newInputIndex, inH, inW, inC);

        // data layout (n, h, k_h, w, k_w, c)
        yGmAddr[index] = bilinearValue;
    }
}

template <typename T>
__aicore__ inline void GridSampler2dBilinearSimt<T>::Process()
{
    uint32_t outImgSize = tiling_->outW * tiling_->outH * tiling_->inC;
    uint32_t shiftB_, mB_, shiftH_, mH_, shiftW_, mW_;
    GetUintDivMagicAndShift(mB_, shiftB_, outImgSize);
    GetUintDivMagicAndShift(mH_, shiftH_, static_cast<uint32_t>(tiling_->outW * tiling_->outH));
    GetUintDivMagicAndShift(mW_, shiftW_, static_cast<uint32_t>(tiling_->outW));
    Simt::VF_CALL<ComputeGridSampler2d<T>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ T*)(inputImgGm_.GetPhyAddr()), (__gm__ T*)(gridGm_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), tiling_->needCoreNum, tiling_->inN, tiling_->inC, tiling_->inH, tiling_->inW,
        tiling_->outH, tiling_->outW, tiling_->paddingMode, tiling_->alignCorners, outImgSize, shiftB_, mB_, shiftH_,
        mH_, shiftW_, mW_, blockId_);
}

} // namespace GridSample
#endif // GRID_SAMPLER_2D_H