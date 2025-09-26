/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _UPSAMPLE_BICUBIC2D_AA_TILING_H_
#define _UPSAMPLE_BICUBIC2D_AA_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct UpsampleBicubic2dAATilingData {
    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;
    float supportH;
    float supportW;
    int32_t maxInterpSizeW;
    int32_t maxInterpSizeH;
    uint32_t radioMatrixWSize;
    uint32_t radioMatrixHSize;
    uint32_t needCoreNumW;
    uint32_t needCoreNumH;
    uint32_t sliceSize;
    uint64_t intermediateMatrixSize;

    int32_t inputShapes[4] = {0};
    int32_t outputShapes[4] = {0};

    int32_t sliceStartListW[50] = {0};
    int32_t sliceEndListW[50] = {0};
    int32_t tailSliceStartListW[50] = {0};
    int32_t tailSliceEndListW[50] = {0};
    int32_t tailRowStartListW[50] = {0};
    int32_t tailRowEndListW[50] = {0};

    int32_t sliceStartListH[50] = {0};
    int32_t sliceEndListH[50] = {0};
    int32_t tailSliceStartListH[50] = {0};
    int32_t tailSliceEndListH[50] = {0};
    int32_t tailBatchStartListH[50] = {0};
    int32_t tailBatchEndListH[50] = {0};

    TCubeTiling matmulTilingW;
    TCubeTiling matmulTilingH;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, UpsampleBicubic2dAATilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (auto i = 0; i < sizeof(UpsampleBicubic2dAATilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, UpsampleBicubic2dAATilingData *constData)
{
    memcpy(constData, tiling, sizeof(UpsampleBicubic2dAATilingData));
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
    UpsampleBicubic2dAATilingData tilingData;  \
    InitTilingData(tilingArg, &tilingData)

#endif