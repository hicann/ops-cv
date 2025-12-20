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
 * \file upsample_trilinear3d_backward_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DEF_H
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct UpsampleTrilinear3dBackwardTilingData {
    uint8_t dataType;
    int64_t batches;
    int64_t inputShapes[3] = {0};
    int64_t outputShapes[3] = {0};

    float scaleW;
    float scaleH;
    float scaleD;
    bool alignCorners;
    bool needResizeW;
    bool needResizeH;
    bool needResizeD;
    int64_t slideSize;
    int64_t radioMatrixSize;
    int64_t intermediateMatrixSizeW;
    int64_t intermediateMatrixSizeH;

    int64_t eachCoreSlideNums[3] = {0};
    int64_t remainders[3] = {0};
    int64_t tailStartSlideNums[3] = {0};
    int64_t groupCoreNums[3] = {0};
    int64_t inputRows[3] = {0};
    int64_t tailAvergingRows[3] = {0};
    int64_t needCoreNums[3] = {0};

    TCubeTiling matmulTilingW;
    TCubeTiling matmulTilingH;
    TCubeTiling matmulTilingD;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, UpsampleTrilinear3dBackwardTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(UpsampleTrilinear3dBackwardTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, UpsampleTrilinear3dBackwardTilingData* constData)
{
    memcpy(constData, tiling, sizeof(UpsampleTrilinear3dBackwardTilingData));
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

#define GET_TILING_DATA(tilingData, tilingArg)        \
    UpsampleTrilinear3dBackwardTilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)

#endif