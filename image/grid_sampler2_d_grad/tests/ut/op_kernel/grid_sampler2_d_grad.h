/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GRID_SAMPLER2D_GRAD_H_
#define GRID_SAMPLER2D_GRAD_H_
#include <vector>

#include "tiling_data_def.h"
#include <map>

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr size_t INTERPOLATION_MODE_INDEX = 0;
constexpr size_t PADDING_MODE_INDEX = 1;
constexpr size_t ALIGN_CORNERS_INDEX = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DTYPE_SIZE_32 = 4;
constexpr int32_t DTYPE_SIZE_16 = 2;
constexpr size_t CHECK_DIM_NUM = 4;
constexpr int BILINEAR = 0;
constexpr int NEAREST = 1;
constexpr int BICUBIC = 2;
constexpr int ZEROS = 0;
constexpr int BORDER = 1;
constexpr int REFLECTION = 2;
constexpr int BILINEAR_DIVIDE_UB_NUM = 54;
constexpr int NEAREST_DIVIDE_UB_NUM = 25;
constexpr uint32_t FP32_GROUP_SIZE_LT_256 = 32;
constexpr uint32_t FP32_GROUP_SIZE_GT_256_LT_512 = 16;
constexpr uint32_t FP32_GROUP_SIZE_GT_512_LT_1024 = 8;
constexpr uint32_t FLOAT_BILINEAR_TILING_KEY = 1;
constexpr uint32_t FLOAT_NEAREST_TILING_KEY = 2;
constexpr uint32_t CHANNEL_256 = 256;
constexpr uint32_t CHANNEL_512 = 512;
constexpr uint32_t CHANNEL_1024 = 1024;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t CONST_TEN = 10;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t RESERVED_UB = 2 * 1024;
constexpr uint32_t RESERVED_UB_CAST = 20 * 1024;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr int CAST_DIVIDE_UB_NUM = 3;
static std::map<std::string, int> INTER_MODE_MAP = {{"bilinear", 0}, {"nearest", 1}, {"bicubic", 2}};
static std::map<std::string, int> PADDING_MODE_MAP = {{"zeros", 0}, {"border", 1}, {"reflection", 2}};
static std::map<bool, int> ALIGN_MODE_MAP = {{true, 1}, {false, 0}};

template <typename TilingData, int32_t dataTypeLen>
class GridSampler2DGradTiling {
public:
    explicit GridSampler2DGradTiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
    {
        this->batch = param.batch;
        this->coreNum = inputCoreNum;
        this->channel = param.channel;
        this->height = param.height;
        this->width = param.width;
        this->gridH = param.gridH;
        this->gridW = param.gridW;
        this->interpolation = param.interpolation;
        this->padding = param.padding;
        this->alignCorners = param.alignCorners;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetUsedCore();
    void GetUsedCoreCast();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b)
    {
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b)
    {
        return (a) / b * b;
    }

private:
    uint32_t batch = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t gridH;
    uint32_t gridW;
    uint32_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t coreNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t group = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 0;
    uint32_t extraUbSize = 0;
    uint32_t usedCoreNumCast = 0;
    uint32_t pNumPerCoreCast = 0;
    uint32_t tailPNumCast = 0;
    uint32_t castElement = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetUsedCore()
{
    uint64_t mulBHW = batch * gridH * gridW;
    if (mulBHW <= this->coreNum) {
        this->usedCoreNum = mulBHW;
        this->pNumPerCore = 1;
        this->tailPNum = 0;
        return;
    }
    this->pNumPerCore = FloorDiv(mulBHW, this->coreNum);
    this->usedCoreNum = this->coreNum;
    this->tailPNum = mulBHW % usedCoreNum;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetUsedCoreCast()
{
    size_t size = batch * channel * height * width;
    if (size <= this->usedCoreNum) {
        this->usedCoreNumCast = size;
        this->pNumPerCoreCast = 1;
        this->tailPNumCast = 0;
        return;
    }
    this->pNumPerCoreCast = FloorDiv(size, this->usedCoreNum);
    this->usedCoreNumCast = this->usedCoreNum;
    this->tailPNumCast = size % usedCoreNumCast;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::SplitUb()
{
    uint32_t alignChannel = CeilAlign(channel, FP32_BLOCK_NUM);
    if (interpolation == 0) {
        divideUbNum = BILINEAR_DIVIDE_UB_NUM;
        extraUbSize = CONST_TEN * alignChannel * DTYPE_SIZE_32;
        group = 1;
    } else if (interpolation == 1) {
        divideUbNum = NEAREST_DIVIDE_UB_NUM;
        if (channel <= CHANNEL_256) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_LT_256 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_LT_256;
        } else if (channel <= CHANNEL_512) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_256_LT_512 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_GT_256_LT_512;
        } else if (channel <= CHANNEL_1024) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_512_LT_1024 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_GT_512_LT_1024;
        } else {
            extraUbSize = BUFFER_NUM * CONST_TWO * alignChannel * DTYPE_SIZE_32;
            group = 1;
        }
    }
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    castElement = (canUseUbSize - RESERVED_UB_CAST) / CAST_DIVIDE_UB_NUM / DTYPE_SIZE_16;
    ubFactorElement = FloorAlign((canUseUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / DTYPE_SIZE_32;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData)
{
    tilingData->batch = batch;
    tilingData->pNumPerCore = pNumPerCore;
    tilingData->tailPNum = tailPNum;
    tilingData->channel = channel;
    tilingData->height = height;
    tilingData->width = width;
    tilingData->gridH = gridH;
    tilingData->gridW = gridW;
    tilingData->blockNum = usedCoreNum;
    tilingData->ubFactorElement = ubFactorElement;
    tilingData->interpolation = interpolation;
    tilingData->padding = padding;
    tilingData->alignCorners = alignCorners;
    tilingData->group = group;
    tilingData->usedCoreNumCast = usedCoreNumCast;
    tilingData->pNumPerCoreCast = pNumPerCoreCast;
    tilingData->tailPNumCast = tailPNumCast;
    tilingData->castElement = castElement;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetTiling(TilingData* tilingData)
{
    GetUsedCore();
    GetUsedCoreCast();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetGridSampler2DGradTiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize)
{
    class GridSampler2DGradTiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}
} // namespace optiling

#endif