/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _UPSAMPLE_BICUBIC2D_TILING_H_
#define _UPSAMPLE_BICUBIC2D_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleBicubic2dTilingData {
    int64_t slide_size;
    int64_t dataType;
    float scale_w;
    float scale_h;
    bool align_corners;

    int16_t max_interp_size_w;
    int16_t max_interp_size_h;
    uint64_t intermediate_matrix_size;
    uint32_t ratio_matrix_size_w;
    uint32_t ratio_matrix_size_h;
    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    int32_t slideStartList_w[MAX_CORE_CONT] = {0};
    int32_t slideEndList_w[MAX_CORE_CONT] = {0};
    int32_t tailSlideStartList_w[MAX_CORE_CONT] = {0};
    int32_t tailSlideEndList_w[MAX_CORE_CONT] = {0};
    int32_t tailRowStartList_w[MAX_CORE_CONT] = {0};
    int32_t tailRowEndList_w[MAX_CORE_CONT] = {0};

    int32_t slideStartList_h[MAX_CORE_CONT] = {0};
    int32_t slideEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailSlideStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailSlideEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailRowStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailRowEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchEndList_h[MAX_CORE_CONT] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, UpsampleBicubic2dTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (auto i = 0; i < sizeof(UpsampleBicubic2dTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, UpsampleBicubic2dTilingData *constData)
{
    memcpy(constData, tiling, sizeof(UpsampleBicubic2dTilingData));
}
#endif

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    UpsampleBicubic2dTilingData tilingData;    \
    InitTilingData(tilingArg, &tilingData)

#endif