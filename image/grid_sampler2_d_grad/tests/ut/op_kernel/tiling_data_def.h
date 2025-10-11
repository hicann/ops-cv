/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

typedef struct {
    uint32_t batch = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t group = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t usedCoreNumCast = 0;
    uint32_t pNumPerCoreCast = 0;
    uint32_t tailPNumCast = 0;
    uint32_t castElement = 0;
} GridSampler2DGradTilingDataTest;

struct InputParamsInfo {
    uint32_t batch;
    uint32_t channel;
    uint32_t height;
    uint32_t width;
    uint32_t gridH;
    uint32_t gridW;
    int interpolation;
    int padding;
    int alignCorners;
};

inline __aicore__ int32_t AlignDiv32(int32_t n)
{
    return ((n + 31) & ~31) / 32;
}

#define COPY_ARR(arrA, arrB, count)        \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i];                 \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#ifdef __CCE_KT_TEST__
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);
#endif

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    do {                                                                             \
        GridSampler2DGradTilingDataTest tilingData;                                          \
        INIT_TILING_DATA(GridSampler2DGradTilingDataTest, tilingDataPointer, tilingPointer); \
        (tilingData).batch = tilingDataPointer->batch;                                   \
        (tilingData).pNumPerCore = tilingDataPointer->pNumPerCore;                       \
        (tilingData).tailPNum = tilingDataPointer->tailPNum;                             \
        (tilingData).channel = tilingDataPointer->channel;                               \
        (tilingData).height = tilingDataPointer->height;                                 \
        (tilingData).width = tilingDataPointer->width;                                   \
        (tilingData).ubFactorElement = tilingDataPointer->ubFactorElement;               \
        (tilingData).blockNum = tilingDataPointer->blockNum;                             \
        (tilingData).interpolation = tilingDataPointer->interpolation;                   \
        (tilingData).padding = tilingDataPointer->padding;                               \
        (tilingData).alignCorners = tilingDataPointer->alignCorners;                     \
        (tilingData).group = tilingDataPointer->group;                                   \
        (tilingData).gridH = tilingDataPointer->gridH;                                   \
        (tilingData).gridW = tilingDataPointer->gridW;                                   \
    } while(0)
#endif // TILING_DATA_DEF_H
