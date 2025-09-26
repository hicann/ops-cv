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
 * \file upsample_nearest3d_grad_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST3D_GRAD_TILING_DEFH
#define UPSAMPLE_NEAREST3D_GRAD_TILING_DEFH

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct UpsampleNearest3dGradTilingData {
    uint8_t dataType;
    int64_t batches;
    float scaleW;
    float scaleH;
    float scaleD;
    bool needResizeW;
    bool needResizeH;
    bool needResizeD;

    int64_t slideSize;
    int64_t tensorSize;
    int64_t tensorSizeMapping;
    int64_t radioMatrixSize;
    int64_t intermediateMatrixSizeW;
    int64_t intermediateMatrixSizeH;

    int64_t gradInputShapes[3] = {0, 0, 0};
    int64_t gradOutputShapes[3] = {0, 0, 0};

    int64_t eachCoreSlideNums[3] = {0, 0, 0};
    int64_t remainders[3] = {0, 0, 0};
    int64_t tailStartSlideNums[3] = {0, 0, 0};
    int64_t groupCoreNums[3] = {0, 0, 0};
    int64_t inputRows[3] = {0, 0, 0};
    int64_t tailAvergingRows[3] = {0, 0, 0};
    int64_t needCoreNums[3] = {0, 0, 0};

    TCubeTiling matmulTilingW;
    TCubeTiling matmulTilingH;
    TCubeTiling matmulTilingD;
};

#define COPY_ARR(arrA, arrB, count)        \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i];                 \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    UpsampleNearest3dGradTilingData tilingData;                                          \
    INIT_TILING_DATA(UpsampleNearest3dGradTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).dataType = tilingDataPointer->dataType;                                 \
    (tilingData).batches = tilingDataPointer->batches;                                   \
    (tilingData).scaleW = tilingDataPointer->scaleW;                                     \
    (tilingData).scaleH = tilingDataPointer->scaleH;                                     \
    (tilingData).scaleD = tilingDataPointer->scaleD;                                     \
    (tilingData).needResizeW = tilingDataPointer->needResizeW;                           \
    (tilingData).needResizeH = tilingDataPointer->needResizeH;                           \
    (tilingData).needResizeD = tilingDataPointer->needResizeD;                           \
    (tilingData).slideSize = tilingDataPointer->slideSize;                               \
    (tilingData).tensorSize = tilingDataPointer->tensorSize;                             \
    (tilingData).tensorSizeMapping = tilingDataPointer->tensorSizeMapping;               \
    (tilingData).radioMatrixSize = tilingDataPointer->radioMatrixSize;                   \
    (tilingData).intermediateMatrixSizeW = tilingDataPointer->intermediateMatrixSizeW;   \
    (tilingData).intermediateMatrixSizeH = tilingDataPointer->intermediateMatrixSizeH;   \
    (tilingData).matmulTilingW = tilingDataPointer->matmulTilingW;                       \
    (tilingData).matmulTilingH = tilingDataPointer->matmulTilingH;                       \
    (tilingData).matmulTilingD = tilingDataPointer->matmulTilingD;                       \
    COPY_ARR((tilingData).gradOutputShapes, tilingDataPointer->gradOutputShapes, 3);     \
    COPY_ARR((tilingData).gradInputShapes, tilingDataPointer->gradInputShapes, 3);       \
    COPY_ARR((tilingData).eachCoreSlideNums, tilingDataPointer->eachCoreSlideNums, 3);   \
    COPY_ARR((tilingData).remainders, tilingDataPointer->remainders, 3);                 \
    COPY_ARR((tilingData).tailStartSlideNums, tilingDataPointer->tailStartSlideNums, 3); \
    COPY_ARR((tilingData).groupCoreNums, tilingDataPointer->groupCoreNums, 3);           \
    COPY_ARR((tilingData).inputRows, tilingDataPointer->inputRows, 3);                   \
    COPY_ARR((tilingData).tailAvergingRows, tilingDataPointer->tailAvergingRows, 3);     \
    COPY_ARR((tilingData).needCoreNums, tilingDataPointer->needCoreNums, 3);

#endif // UPSAMPLE_NEAREST3D_GRAD_TILING_DEFH