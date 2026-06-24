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
 * \file grid_sampler_3d_bilinear_smit.h
 * \brief grid sampler 3d kernel info
 */
#ifndef GRID_SAMPLER_3D_BILINEAR_SIMT_H
#define GRID_SAMPLER_3D_BILINEAR_SIMT_H
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "grid_sampler_bilinear_smit_common.h"

namespace GridSample {

using namespace AscendC;

template <typename T, typename T_IDX, typename U_IDX>
class GridSampler3dBilinearSimt {
public:
    __aicore__ inline GridSampler3dBilinearSimt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace,
                                const GridSampleTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputImgGm_;
    GlobalTensor<T> gridGm_;
    GlobalTensor<T> yGm_;
    uint32_t blockId_ = GetBlockIdx();
    const GridSampleTilingData* tiling_;
};

template <typename T, typename T_IDX, typename U_IDX>
__aicore__ inline void GridSampler3dBilinearSimt<T, T_IDX, U_IDX>::Init(
    GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData* __restrict tilingData)
{
    inputImgGm_.SetGlobalBuffer((__gm__ T*)(x));
    gridGm_.SetGlobalBuffer((__gm__ T*)(grid));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    tiling_ = tilingData;
}

template <typename T, typename T_IDX, typename U_IDX>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline T GetInputPointValue(
    __gm__ T* inputImgGmAddr, U_IDX inputDepth, U_IDX inputHeight, U_IDX inputWidth, T_IDX channelIndex,
    T_IDX inputDataBatchOffset, T_IDX inD, T_IDX inH, T_IDX inW, T_IDX inC)
{
    if (inputDepth >= 0 && inputHeight >= 0 && inputWidth >= 0 && inputDepth < inD && inputHeight < inH &&
        inputWidth < inW) {
        return inputImgGmAddr[inputDataBatchOffset + inputDepth * inH * inW + inputHeight * inW + inputWidth +
                              channelIndex * inH * inW * inD];
    }
    return static_cast<T>(0.0);
}

template <typename T, typename T_IDX, typename U_IDX>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline T ComputeBilinear(
    __gm__ T* inputImgGmAddr, float pointDepth, float pointHeight, float pointWidth, T_IDX channelIndex,
    T_IDX inputDataBatchOffset, T_IDX inD, T_IDX inH, T_IDX inW, T_IDX inC, T_IDX index)
{
    float depthFloor = floorf(pointDepth);
    float heightFloor = floorf(pointHeight);
    float widthFloor = floorf(pointWidth);

    float depthFloorDelta = pointDepth - depthFloor;
    float heightFloorDelta = pointHeight - heightFloor;
    float widthFloorDelta = pointWidth - widthFloor;

    // pointFrontLeftUp
    float inputValue = static_cast<float>(
        GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, depthFloor, heightFloor, widthFloor,
                                            channelIndex, inputDataBatchOffset, inD, inH, inW, inC));
    float inputWeight = (1.0f - widthFloorDelta) * (1.0f - heightFloorDelta) * (1.0f - depthFloorDelta);
    float bilinearValue = (inputValue * inputWeight);

    // pointFrontRightUp
    inputValue = static_cast<float>(GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, depthFloor,
                                                                        heightFloor, (widthFloor + 1), channelIndex,
                                                                        inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = widthFloorDelta * (1.0f - heightFloorDelta) * (1.0f - depthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointFrontLeftBottom
    inputValue = static_cast<float>(GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, depthFloor,
                                                                        (heightFloor + 1), widthFloor, channelIndex,
                                                                        inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = (1.0f - widthFloorDelta) * heightFloorDelta * (1.0f - depthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointFrontRightBottom
    inputValue = static_cast<float>(
        GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, depthFloor, (heightFloor + 1), (widthFloor + 1),
                                            channelIndex, inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = widthFloorDelta * heightFloorDelta * (1.0f - depthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointBackLeftUp
    inputValue = static_cast<float>(GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, (depthFloor + 1),
                                                                        heightFloor, widthFloor, channelIndex,
                                                                        inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = (1.0f - widthFloorDelta) * (1.0f - heightFloorDelta) * depthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointBackRightUp
    inputValue = static_cast<float>(GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, (depthFloor + 1),
                                                                        heightFloor, (widthFloor + 1), channelIndex,
                                                                        inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = widthFloorDelta * (1.0f - heightFloorDelta) * depthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointBackLeftBottom
    inputValue = static_cast<float>(GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, (depthFloor + 1),
                                                                        (heightFloor + 1), widthFloor, channelIndex,
                                                                        inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = (1.0f - widthFloorDelta) * heightFloorDelta * depthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointBackRightBottom
    inputValue = static_cast<float>(
        GetInputPointValue<T, T_IDX, U_IDX>((__gm__ T*)inputImgGmAddr, (depthFloor + 1), (heightFloor + 1),
                                            (widthFloor + 1), channelIndex, inputDataBatchOffset, inD, inH, inW, inC));
    inputWeight = widthFloorDelta * heightFloorDelta * depthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    return static_cast<T>(bilinearValue);
}

// LAUNCH_BOUND
template <typename T, typename T_IDX, typename U_IDX>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM_3D) __aicore__
    void ComputeGridSampler3d(__gm__ T* inputImgGmAddr, __gm__ T* gridGmAddr, __gm__ T* yGmAddr, int32_t blockNum,
                              T_IDX intN, T_IDX inC, T_IDX inD, T_IDX inH, T_IDX inW, T_IDX outD, T_IDX outH,
                              T_IDX outW, int32_t paddingMode, int32_t alignCorners, T_IDX outImgSize, T_IDX shiftB_,
                              T_IDX mB_, T_IDX shiftD_, T_IDX mD_, T_IDX shiftH_, T_IDX mH_, T_IDX shiftW_, T_IDX mW_,
                              T_IDX blockId_)
{
    for (T_IDX index = blockId_ * VF_MAX_THREAD_NUM_3D + threadIdx.x; index < outImgSize * intN;
         index += (blockNum * VF_MAX_THREAD_NUM_3D)) {
        // output info (N H K_h W K_w, groups, groupC)
        T_IDX batchNum, depthCol, heightCol, widthCol, channelIndex;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(index, mB_, shiftB_);
        T_IDX remain = index - batchNum * outImgSize;

        channelIndex = Simt::UintDiv(remain, mD_, shiftD_);
        remain = remain - channelIndex * (outW * outH * outD);

        depthCol = Simt::UintDiv(remain, mH_, shiftH_);
        remain = remain - depthCol * (outW * outH);

        heightCol = Simt::UintDiv(remain, mW_, shiftW_);
        widthCol = remain - heightCol * outW;

        T_IDX newInputIndex = batchNum * inD * inH * inW * inC;
        T_IDX offsetBaseAddr = batchNum * outD * outH * outW * 3 + depthCol * outH * outW * 3 + heightCol * outW * 3 +
                               widthCol * 3;

        // offset height info
        float gridDepthValue = static_cast<float>(gridGmAddr[offsetBaseAddr + 2]);
        float gridHeigthValue = static_cast<float>(gridGmAddr[offsetBaseAddr + 1]);
        float gridWeightValue = static_cast<float>(gridGmAddr[offsetBaseAddr]);
        if (alignCorners == 1) {
            gridDepthValue = (gridDepthValue + 1) / 2 * (inD - 1);
            gridHeigthValue = (gridHeigthValue + 1) / 2 * (inH - 1);
            gridWeightValue = (gridWeightValue + 1) / 2 * (inW - 1);
        } else {
            gridDepthValue = ((gridDepthValue + 1) * inD - 1) / 2;
            gridHeigthValue = ((gridHeigthValue + 1) * inH - 1) / 2;
            gridWeightValue = ((gridWeightValue + 1) * inW - 1) / 2;
        }

        gridDepthValue = Clip<T_IDX>(gridDepthValue, inD, paddingMode, alignCorners);
        gridWeightValue = Clip<T_IDX>(gridWeightValue, inW, paddingMode, alignCorners);
        gridHeigthValue = Clip<T_IDX>(gridHeigthValue, inH, paddingMode, alignCorners);

        T bilinearValue = ComputeBilinear<T, T_IDX, U_IDX>((__gm__ T*)(inputImgGmAddr), gridDepthValue, gridHeigthValue,
                                                           gridWeightValue, channelIndex, newInputIndex, inD, inH, inW,
                                                           inC, index);

        // data layout (n, h, k_h, w, k_w, c)
        yGmAddr[index] = bilinearValue;
    }
}

template <typename T, typename T_IDX, typename U_IDX>
__aicore__ inline void GridSampler3dBilinearSimt<T, T_IDX, U_IDX>::Process()
{
    T_IDX outImgSize = tiling_->outD * tiling_->outW * tiling_->outH * tiling_->inC;
    T_IDX shiftB_, mB_, shiftD_, mD_, shiftH_, mH_, shiftW_, mW_;
    GetUintDivMagicAndShift(mB_, shiftB_, outImgSize);
    GetUintDivMagicAndShift(mD_, shiftD_, static_cast<T_IDX>(tiling_->outW * tiling_->outH * tiling_->outD));
    GetUintDivMagicAndShift(mH_, shiftH_, static_cast<T_IDX>(tiling_->outW * tiling_->outH));
    GetUintDivMagicAndShift(mW_, shiftW_, static_cast<T_IDX>(tiling_->outW));
    asc_vf_call<ComputeGridSampler3d<T, T_IDX, U_IDX>>(
        dim3{VF_MAX_THREAD_NUM_3D, 1, 1}, (__gm__ T*)(inputImgGm_.GetPhyAddr()), (__gm__ T*)(gridGm_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), tiling_->needCoreNum, tiling_->inN, tiling_->inC, tiling_->inD, tiling_->inH,
        tiling_->inW, tiling_->outD, tiling_->outH, tiling_->outW, tiling_->paddingMode, tiling_->alignCorners,
        outImgSize, shiftB_, mB_, shiftD_, mD_, shiftH_, mH_, shiftW_, mW_, blockId_);
}

} // namespace GridSample
#endif // GRID_SAMPLER_3D_H