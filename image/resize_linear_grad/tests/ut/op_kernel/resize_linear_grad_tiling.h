/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_linear_grad_tiling.h
 * \brief resize_linear_grad tiling data definition for kernel UT
 */

#ifndef RESIZE_LINEAR_GRAD_TILING_DEFH
#define RESIZE_LINEAR_GRAD_TILING_DEFH

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

// Define struct directly in UT directory (no dependency on BEGIN_TILING_DATA_DEF macro)
struct ResizeLinearGradTilingData {
    int64_t realCoreNum;
    int64_t initCoreNum;
    int64_t blkProcessNum;
    int64_t ubLoopSizeB;
    int64_t ubLoopSizeT;
    int64_t ubFactor;
    int64_t ubFactorTailB;
    int64_t ubFactorTailT;
    int64_t lenSrcLOrUb;
    int64_t lenDesL;
    float scaleL;
    float inverseScaleL;
};

#define __aicore__
#ifdef __NPU_TILING__
inline __aicore__ void InitTilingData(const __gm__ uint8_t* tiling, ResizeLinearGradTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(ResizeLinearGradTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, ResizeLinearGradTilingData* constData)
{
    memcpy(constData, tiling, sizeof(ResizeLinearGradTilingData));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        (__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    ResizeLinearGradTilingData tilingData;     \
    InitTilingData(tilingArg, &tilingData)

#define REGISTER_TILING_DEFAULT(T)

#endif // RESIZE_LINEAR_GRAD_TILING_DEFH