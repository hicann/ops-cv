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
 * \file upsample_bilinear2d_aa_tiling_def.h
 * \brief
 */

#ifndef UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H
#define UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 192;

struct UpsampleLinear1dTilingData {
    int64_t mode;
    bool align_corners;
    int64_t slide_size_w;
    int64_t slide_size_h;
    int64_t dataType;
    float scale_w;
    float scale_h;

    uint64_t intermediate_matrix_size;
    uint32_t radio_matrix_size_w;
    uint32_t radio_matrix_size_h;
    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t eachCoreSlideNumH;
    int64_t tailStartSlideNumH;
    int64_t slideNumH;
    int64_t groupCoreNumH;
    int64_t tailAvergingRowsH;
    int64_t remainderH;

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
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

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    UpsampleLinear1dTilingData tilingData;                                               \
    INIT_TILING_DATA(UpsampleLinear1dTilingData, tilingDataPointer, tilingPointer);      \
    (tilingData).mode = tilingDataPointer->mode;                                         \
    (tilingData).align_corners = tilingDataPointer->align_corners;                       \
    (tilingData).slide_size_w = tilingDataPointer->slide_size_w;                         \
    (tilingData).slide_size_h = tilingDataPointer->slide_size_h;                         \
    (tilingData).dataType = tilingDataPointer->dataType;                                 \
    (tilingData).scale_w = tilingDataPointer->scale_w;                                   \
    (tilingData).scale_h = tilingDataPointer->scale_h;                                   \
    (tilingData).intermediate_matrix_size = tilingDataPointer->intermediate_matrix_size; \
    (tilingData).radio_matrix_size_w = tilingDataPointer->radio_matrix_size_w;           \
    (tilingData).radio_matrix_size_h = tilingDataPointer->radio_matrix_size_h;           \
    (tilingData).need_core_num_w = tilingDataPointer->need_core_num_w;                   \
    (tilingData).need_core_num_h = tilingDataPointer->need_core_num_h;                   \
    (tilingData).eachCoreSlideNumW = tilingDataPointer->eachCoreSlideNumW;               \
    (tilingData).tailStartSlideNumW = tilingDataPointer->tailStartSlideNumW;             \
    (tilingData).slideNumW = tilingDataPointer->slideNumW;                               \
    (tilingData).groupCoreNumW = tilingDataPointer->groupCoreNumW;                       \
    (tilingData).tailAvergingRowsW = tilingDataPointer->tailAvergingRowsW;               \
    (tilingData).remainderW = tilingDataPointer->remainderW;                             \
    (tilingData).eachCoreSlideNumH = tilingDataPointer->eachCoreSlideNumH;               \
    (tilingData).tailStartSlideNumH = tilingDataPointer->tailStartSlideNumH;             \
    (tilingData).slideNumH = tilingDataPointer->slideNumH;                               \
    (tilingData).groupCoreNumH = tilingDataPointer->groupCoreNumH;                       \
    (tilingData).tailAvergingRowsH = tilingDataPointer->tailAvergingRowsH;               \
    (tilingData).remainderH = tilingDataPointer->remainderH;                             \
    (tilingData).matmulTiling_w = tilingDataPointer->matmulTiling_w;                     \
    (tilingData).matmulTiling_h = tilingDataPointer->matmulTiling_h;                     \
    COPY_ARR((tilingData).input_shapes, tilingDataPointer->input_shapes, 4);             \
    COPY_ARR((tilingData).output_shapes, tilingDataPointer->output_shapes, 4);

#endif  // UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H