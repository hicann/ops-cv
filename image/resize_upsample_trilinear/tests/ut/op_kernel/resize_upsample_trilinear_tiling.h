/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UPSAMPLE_TRILINEAR3D_TILING_H_
#define UPSAMPLE_TRILINEAR3D_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct UpsampleTrilinearTilingData {
    float scale_w;
    float scale_h;
    float scale_d;
    uint16_t total_core_num;
    uint64_t ratio_metrix_size;
    int64_t output_w;
    int64_t output_h;
    int64_t output_d;
    int64_t input_w;
    int64_t input_h;
    int64_t input_d;
    int64_t batches;
    uint32_t align_corners;

    uint32_t tail_group_start_inx_w_list[48] = {0};
    uint32_t tail_group_end_inx_w_list[48] = {0};
    uint32_t tail_group_slide_start_inx_w_list[48] = {0};
    uint32_t tail_group_slide_end_inx_w_list[48] = {0};

    uint32_t tail_group_start_inx_h_list[48] = {0};
    uint32_t tail_group_end_inx_h_list[48] = {0};
    uint32_t tail_group_batch_start_inx_h_list[48] = {0};
    uint32_t tail_group_batch_end_inx_h_list[48] = {0};

    uint32_t tail_group_start_inx_d_list[48] = {0};
    uint32_t tail_group_end_inx_d_list[48] = {0};
    uint32_t tail_group_batch_start_inx_d_list[48] = {0};
    uint32_t tail_group_batch_end_inx_d_list[48] = {0};

    TCubeTiling matmul_tiling_w;
    TCubeTiling matmul_tiling_h;
    TCubeTiling matmul_tiling_d;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, UpsampleTrilinearTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(UpsampleTrilinearTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, UpsampleTrilinearTilingData* constData)
{
    memcpy(constData, tiling, sizeof(UpsampleTrilinearTilingData));
}
#endif

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    UpsampleTrilinearTilingData tilingData;    \
    InitTilingData(tilingArg, &tilingData)

#endif