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
 * \file upsample_nearest_exact2d_grad_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST_EXACT2D_GRAD_TILING_DEF_H
#define UPSAMPLE_NEAREST_EXACT2D_GRAD_TILING_DEF_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleNearestExact2dGradTilingData {
    int64_t slide_size;
    uint8_t dataType;
    float scale_w;
    float scale_h;
    float invscale_w;
    float invscale_h;

    float support_w;
    float support_h;
    int16_t max_interp_size_w;
    int16_t max_interp_size_h;
    uint64_t intermediate_matrix_size;
    uint32_t radio_matrix_size;
    uint32_t radio_matrix_size_h;
    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t input_shapes[4] = {1, 1, 4, 4};
    int64_t output_shapes[4] = {1, 1, 16, 16};

    int64_t slideStartList_w[MAX_CORE_CONT] = {0};
    int64_t slideEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_w[MAX_CORE_CONT] = {0};

    int64_t slideStartList_h[MAX_CORE_CONT] = {0};
    int64_t slideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailBatchStartListH[MAX_CORE_CONT] = {0};
    int64_t tailBatchEndListH[MAX_CORE_CONT] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
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

#define GET_TILING_DATA(tilingData, tilingPointer)                                                       \
    UpsampleNearestExact2dGradTilingData tilingData;                                                     \
    INIT_TILING_DATA(UpsampleNearestExact2dGradTilingData, tilingDataPointer, tilingPointer);            \
    (tilingData).slide_size = tilingDataPointer->slide_size;                                             \
    (tilingData).dataType = tilingDataPointer->dataType;                                                 \
    (tilingData).scale_w = tilingDataPointer->scale_w;                                                   \
    (tilingData).scale_h = tilingDataPointer->scale_h;                                                   \
    (tilingData).invscale_w = tilingDataPointer->invscale_w;                                             \
    (tilingData).invscale_h = tilingDataPointer->invscale_h;                                             \
    (tilingData).support_w = tilingDataPointer->support_w;                                               \
    (tilingData).max_interp_size_w = tilingDataPointer->max_interp_size_w;                               \
    (tilingData).support_h = tilingDataPointer->support_h;                                               \
    (tilingData).max_interp_size_h = tilingDataPointer->max_interp_size_h;                               \
    (tilingData).intermediate_matrix_size = tilingDataPointer->intermediate_matrix_size;                 \
    (tilingData).radio_matrix_size = tilingDataPointer->radio_matrix_size;                               \
    (tilingData).radio_matrix_size_h = tilingDataPointer->radio_matrix_size_h;                           \
    (tilingData).need_core_num_w = tilingDataPointer->need_core_num_w;                                   \
    (tilingData).need_core_num_h = tilingDataPointer->need_core_num_h;                                   \
    (tilingData).matmulTiling_w = tilingDataPointer->matmulTiling_w;                                     \
    (tilingData).matmulTiling_h = tilingDataPointer->matmulTiling_h;                                     \
    COPY_ARR((tilingData).input_shapes, tilingDataPointer->input_shapes, 4);                             \
    COPY_ARR((tilingData).output_shapes, tilingDataPointer->output_shapes, 4);                           \
    COPY_ARR((tilingData).slideStartList_w, tilingDataPointer->slideStartList_w, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideEndList_w, tilingDataPointer->slideEndList_w, MAX_CORE_CONT);             \
    COPY_ARR((tilingData).tailSlideStartList_w, tilingDataPointer->tailSlideStartList_w, MAX_CORE_CONT); \
    COPY_ARR((tilingData).tailSlideEndList_w, tilingDataPointer->tailSlideEndList_w, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowStartList_w, tilingDataPointer->tailRowStartList_w, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowEndList_w, tilingDataPointer->tailRowEndList_w, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideStartList_h, tilingDataPointer->slideStartList_h, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).slideEndList_h, tilingDataPointer->slideEndList_h, MAX_CORE_CONT);             \
    COPY_ARR((tilingData).tailSlideStartList_h, tilingDataPointer->tailSlideStartList_h, MAX_CORE_CONT); \
    COPY_ARR((tilingData).tailSlideEndList_h, tilingDataPointer->tailSlideEndList_h, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowStartList_h, tilingDataPointer->tailRowStartList_h, MAX_CORE_CONT);     \
    COPY_ARR((tilingData).tailRowEndList_h, tilingDataPointer->tailRowEndList_h, MAX_CORE_CONT);         \
    COPY_ARR((tilingData).tailBatchStartListH, tilingDataPointer->tailBatchStartListH, MAX_CORE_CONT);   \
    COPY_ARR((tilingData).tailBatchEndListH, tilingDataPointer->tailBatchEndListH, MAX_CORE_CONT);

#endif // UPSAMPLE_NEAREST_EXACT2D_GRAD_TILING_DEF_H