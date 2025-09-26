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
 * \file upsample_bilinear2d_aa_backward_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DEFH
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DEFH

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleBilinear2dAABackwardTilingData {
    int64_t slideSize;
    uint8_t dataType;
    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;

    float supportW;
    float supportH;
    int16_t maxInterpSizeW;
    int16_t maxInterpSizeH;
    uint64_t intermediateMatrixSize;
    uint32_t radioMatrixSizeW;
    uint32_t radioMatrixSizeH;
    uint32_t needCoreNumW;
    uint32_t needCoreNumH;
    bool needResizeW;
    bool needResizeH;

    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};

    int64_t slideStartListW[MAX_CORE_CONT] = {0};
    int64_t slideEndListW[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartListW[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndListW[MAX_CORE_CONT] = {0};
    int64_t tailRowStartListW[MAX_CORE_CONT] = {0};
    int64_t tailRowEndListW[MAX_CORE_CONT] = {0};

    int64_t slideStartListH[MAX_CORE_CONT] = {0};
    int64_t slideEndListH[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartListH[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndListH[MAX_CORE_CONT] = {0};
    int64_t tailRowStartListH[MAX_CORE_CONT] = {0};
    int64_t tailRowEndListH[MAX_CORE_CONT] = {0};

    TCubeTiling matmulTilingW;
    TCubeTiling matmulTilingH;
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

#define GET_TILING_DATA(tilingData, tilingPointer)                                                     \
    UpsampleBilinear2dAABackwardTilingData tilingData;                                                 \
    INIT_TILING_DATA(UpsampleBilinear2dAABackwardTilingData, tilingDataPointer, tilingPointer);        \
    (tilingData).slideSize = tilingDataPointer->slideSize;                                             \
    (tilingData).dataType = tilingDataPointer->dataType;                                               \
    (tilingData).scaleW = tilingDataPointer->scaleW;                                                   \
    (tilingData).scaleH = tilingDataPointer->scaleH;                                                   \
    (tilingData).invscaleW = tilingDataPointer->invscaleW;                                             \
    (tilingData).invscaleH = tilingDataPointer->invscaleH;                                             \
    (tilingData).supportW = tilingDataPointer->supportW;                                               \
    (tilingData).maxInterpSizeW = tilingDataPointer->maxInterpSizeW;                                   \
    (tilingData).supportH = tilingDataPointer->supportH;                                               \
    (tilingData).maxInterpSizeH = tilingDataPointer->maxInterpSizeH;                                   \
    (tilingData).intermediateMatrixSize = tilingDataPointer->intermediateMatrixSize;                   \
    (tilingData).radioMatrixSizeW = tilingDataPointer->radioMatrixSizeW;                               \
    (tilingData).radioMatrixSizeH = tilingDataPointer->radioMatrixSizeH;                               \
    (tilingData).needCoreNumW = tilingDataPointer->needCoreNumW;                                       \
    (tilingData).needCoreNumH = tilingDataPointer->needCoreNumH;                                       \
    (tilingData).needResizeW = tilingDataPointer->needResizeW;                                         \
    (tilingData).needResizeH = tilingDataPointer->needResizeH;                                         \
    (tilingData).matmulTilingW = tilingDataPointer->matmulTilingW;                                     \
    (tilingData).matmulTilingH = tilingDataPointer->matmulTilingH;                                     \
    COPY_ARR((tilingData).inputShapes, tilingDataPointer->inputShapes, 4);                             \
    COPY_ARR((tilingData).outputShapes, tilingDataPointer->outputShapes, 4);                           \
    COPY_ARR((tilingData).slideStartListW, tilingDataPointer->slideStartListW, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideEndListW, tilingDataPointer->slideEndListW, MAX_CORE_CONT);             \
    COPY_ARR((tilingData).tailSlideStartListW, tilingDataPointer->tailSlideStartListW, MAX_CORE_CONT); \
    COPY_ARR((tilingData).tailSlideEndListW, tilingDataPointer->tailSlideEndListW, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowStartListW, tilingDataPointer->tailRowStartListW, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowEndListW, tilingDataPointer->tailRowEndListW, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideStartListH, tilingDataPointer->slideStartListH, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideEndListH, tilingDataPointer->slideEndListH, MAX_CORE_CONT);             \
    COPY_ARR((tilingData).tailSlideStartListH, tilingDataPointer->tailSlideStartListH, MAX_CORE_CONT); \
    COPY_ARR((tilingData).tailSlideEndListH, tilingDataPointer->tailSlideEndListH, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowStartListH, tilingDataPointer->tailRowStartListH, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowEndListH, tilingDataPointer->tailRowEndListH, MAX_CORE_CONT);

#endif  // UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DEFH