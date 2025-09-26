/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest3d_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST3D_TILING_DEFH
#define UPSAMPLE_NEAREST3D_TILING_DEFH

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct UpsampleNearest3dTilingData {
    uint8_t dataType;
    int64_t batches;
    int64_t inputShapes[3] = {0, 0, 0};
    int64_t outputShapes[3] = {0, 0, 0};

    float scaleW;
    float scaleH;
    float scaleD;
    int64_t slideSizeW;
    int64_t tensorSizeW;
    int64_t tensorSizeH;
    int64_t tensorSizeD;

    int64_t slideNumH;
    int64_t slideNumD;
    int64_t eachCoreSlideNum;
    int64_t remainder;
    int64_t tailStartSlideNum;
    int64_t groupCoreNum;
    int64_t inputRow;
    int64_t tailAvergingRow;
    int64_t needCoreNum;
};

#define COPY_ARR(arrA, arrB, count)        \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i];                 \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    UpsampleNearest3dTilingData tilingData;                                          \
    INIT_TILING_DATA(UpsampleNearest3dTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).dataType = tilingDataPointer->dataType;                             \
    (tilingData).batches = tilingDataPointer->batches;                               \
    (tilingData).scaleW = tilingDataPointer->scaleW;                                 \
    (tilingData).scaleH = tilingDataPointer->scaleH;                                 \
    (tilingData).scaleD = tilingDataPointer->scaleD;                                 \
    (tilingData).slideSizeW = tilingDataPointer->slideSizeW;                         \
    (tilingData).tensorSizeW = tilingDataPointer->tensorSizeW;                       \
    (tilingData).tensorSizeH = tilingDataPointer->tensorSizeH;                       \
    (tilingData).tensorSizeD = tilingDataPointer->tensorSizeD;                       \
    (tilingData).slideNumH = tilingDataPointer->slideNumH;                           \
    (tilingData).slideNumD = tilingDataPointer->slideNumD;                           \
    (tilingData).eachCoreSlideNum = tilingDataPointer->eachCoreSlideNum;             \
    (tilingData).remainder = tilingDataPointer->remainder;                           \
    (tilingData).tailStartSlideNum = tilingDataPointer->tailStartSlideNum;           \
    (tilingData).groupCoreNum = tilingDataPointer->groupCoreNum;                     \
    (tilingData).inputRow = tilingDataPointer->inputRow;                             \
    (tilingData).tailAvergingRow = tilingDataPointer->tailAvergingRow;               \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;                       \
    COPY_ARR((tilingData).inputShapes, tilingDataPointer->inputShapes, 3);           \
    COPY_ARR((tilingData).outputShapes, tilingDataPointer->outputShapes, 3);

#endif  // UPSAMPLE_NEAREST3D_TILING_DEFH