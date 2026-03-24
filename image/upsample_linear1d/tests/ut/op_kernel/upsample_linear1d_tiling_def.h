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
 * \file upsample_linear1d_tiling_def.h
 * \brief
 */

#ifndef UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H
#define UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 192;

struct UpsampleLinear1dTilingData {
    bool align_corners;
    int64_t slide_size_w;
    float scale_w;
    uint32_t radio_matrix_size_w;
    uint32_t need_core_num_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    uint64_t mPerTime;
    uint64_t loopTimes0;
    uint64_t loopTimes1;
    uint64_t loopTail0;
    uint64_t loopTail1;
    uint64_t inputUbSize;
    uint64_t outputUbSize;

    uint64_t loopTailTimes0;
    uint64_t loopTailTimes1;
    uint64_t loopTailTail0;
    uint64_t loopTailTail1;

    uint64_t matmulLoopTimes;
    uint64_t matmulBlockTail;
    uint64_t matmulBlockTail0;

    uint64_t remainderMatmulLoopTimes;
    uint64_t remainderMatmulBlockTail;
    uint64_t remainderMatmulBlockTail0;
    uint64_t remainderLoopTailTimes0;
    uint64_t remainderLoopTailTimes1;
    uint64_t remainderLoopTailTail0;
    uint64_t remainderLoopTailTail1;

    uint64_t matmulBlockPerTime;
    uint64_t matmulBlockPerTime0;
    uint64_t inputH;
    uint64_t mmInputNum;
    uint64_t mmtotalPerCoreNum;
    uint64_t blockSizeNum;

    int64_t input_shapes[3] = {0, 0, 0, 0};
    int64_t output_shapes[3] = {0, 0, 0, 0};

    TCubeTiling matmulTiling_w;
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
    (tilingData).align_corners = tilingDataPointer->align_corners;                       \
    (tilingData).slide_size_w = tilingDataPointer->slide_size_w;                         \
    (tilingData).scale_w = tilingDataPointer->scale_w;                                   \
    (tilingData).radio_matrix_size_w = tilingDataPointer->radio_matrix_size_w;           \
    (tilingData).need_core_num_w = tilingDataPointer->need_core_num_w;                   \
    (tilingData).eachCoreSlideNumW = tilingDataPointer->eachCoreSlideNumW;               \
    (tilingData).tailStartSlideNumW = tilingDataPointer->tailStartSlideNumW;             \
    (tilingData).slideNumW = tilingDataPointer->slideNumW;                               \
    (tilingData).groupCoreNumW = tilingDataPointer->groupCoreNumW;                       \
    (tilingData).tailAvergingRowsW = tilingDataPointer->tailAvergingRowsW;               \
    (tilingData).remainderW = tilingDataPointer->remainderW;                             \
    (tilingData).matmulTiling_w = tilingDataPointer->matmulTiling_w;                     \
    (tilingData).mPerTime = tilingDataPointer->mPerTime;                                 \
    (tilingData).loopTimes0 = tilingDataPointer->loopTimes0;                               \
    (tilingData).loopTimes1 = tilingDataPointer->loopTimes1;                               \
    (tilingData).loopTail0 = tilingDataPointer->loopTail0;                                 \
    (tilingData).loopTail1 = tilingDataPointer->loopTail1;                                 \
    (tilingData).inputUbSize = tilingDataPointer->inputUbSize;                           \
    (tilingData).outputUbSize = tilingDataPointer->outputUbSize;                         \
    (tilingData).matmulLoopTimes = tilingDataPointer->matmulLoopTimes;                         \
    (tilingData).matmulBlockTail = tilingDataPointer->matmulBlockTail;                                 \
    (tilingData).matmulBlockTail0 = tilingDataPointer->matmulBlockTail0;                                 \
    (tilingData).matmulBlockPerTime = tilingDataPointer->matmulBlockPerTime;                               \
    (tilingData).matmulBlockPerTime0 = tilingDataPointer->matmulBlockPerTime0;                               \
    (tilingData).loopTailTimes0 = tilingDataPointer->loopTailTimes0;                                 \
    (tilingData).loopTailTimes1 = tilingDataPointer->loopTailTimes1;                                 \
    (tilingData).loopTailTail0 = tilingDataPointer->loopTailTail0;                               \
    (tilingData).loopTailTail1 = tilingDataPointer->loopTailTail1;                               \
    (tilingData).remainderMatmulLoopTimes = tilingDataPointer->remainderMatmulLoopTimes;                               \
    (tilingData).remainderMatmulBlockTail = tilingDataPointer->remainderMatmulBlockTail;                                 \
    (tilingData).remainderMatmulBlockTail0 = tilingDataPointer->remainderMatmulBlockTail0;                                 \
    (tilingData).remainderLoopTailTimes0 = tilingDataPointer->remainderLoopTailTimes0;                               \
    (tilingData).remainderLoopTailTimes1 = tilingDataPointer->remainderLoopTailTimes1;                               \
    (tilingData).remainderLoopTailTail0 = tilingDataPointer->remainderLoopTailTail0;                               \
    (tilingData).remainderLoopTailTail1 = tilingDataPointer->remainderLoopTailTail1;                               \
    (tilingData).inputH = tilingDataPointer->inputH;                               \
    (tilingData).mmInputNum = tilingDataPointer->mmInputNum;                               \
    (tilingData).mmtotalPerCoreNum = tilingDataPointer->mmtotalPerCoreNum;                               \
    (tilingData).blockSizeNum = tilingDataPointer->blockSizeNum;                               \
    COPY_ARR((tilingData).input_shapes, tilingDataPointer->input_shapes, 3);             \
    COPY_ARR((tilingData).output_shapes, tilingDataPointer->output_shapes, 3);

#endif  // UPSAMPLE_LINEAR1D_INPLACE_TILING_DEF_H