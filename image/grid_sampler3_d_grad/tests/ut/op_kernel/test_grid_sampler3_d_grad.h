/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GRID_SAMPLE_TILING_H_
#define GRID_SAMPLE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 200

#pragma pack(1)

struct GridSampler3DGradTilingDataTest {
    int64_t batch = 0;
    int64_t channel = 0;
    int64_t xD = 0;
    int64_t xH = 0;
    int64_t xW = 0;
    int64_t gridD = 0;
    int64_t gridH = 0;
    int64_t gridW = 0;
    int64_t interpolation = 0;
    int64_t padding = 0;
    int64_t alignCorners = 0;
    int64_t blockNum = 0;
    int64_t pNumPerCore = 0;
    int64_t dxNumPerCore = 0;
    int64_t tailPNum = 0;
    int64_t group = 0;
    int64_t ubFactorElement = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    GridSampler3DGradTilingDataTest tilingData;                                          \
    INIT_TILING_DATA(GridSampler3DGradTilingDataTest, tilingDataPointer, tilingPointer); \
    (tilingData).batch = tilingDataPointer->batch;                                   \
    (tilingData).channel = tilingDataPointer->channel;                               \
    (tilingData).xD = tilingDataPointer->xD;                                         \
    (tilingData).xH = tilingDataPointer->xH;                                         \
    (tilingData).xW = tilingDataPointer->xW;                                         \
    (tilingData).gridD = tilingDataPointer->gridD;                                   \
    (tilingData).gridH = tilingDataPointer->gridH;                                   \
    (tilingData).gridW = tilingDataPointer->gridW;                                   \
    (tilingData).interpolation = tilingDataPointer->interpolation;                   \
    (tilingData).padding = tilingDataPointer->padding;                               \
    (tilingData).alignCorners = tilingDataPointer->alignCorners;                     \
    (tilingData).blockNum = tilingDataPointer->blockNum;                             \
    (tilingData).pNumPerCore = tilingDataPointer->pNumPerCore;                       \
    (tilingData).dxNumPerCore = tilingDataPointer->dxNumPerCore;                     \
    (tilingData).tailPNum = tilingDataPointer->tailPNum;                             \
    (tilingData).group = tilingDataPointer->group;                                   \
    (tilingData).ubFactorElement = tilingDataPointer->ubFactorElement;
#endif