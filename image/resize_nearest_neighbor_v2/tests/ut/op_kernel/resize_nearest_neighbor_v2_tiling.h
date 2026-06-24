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
 * \file resize_nearest_neighbor_v2_tiling.h
 * \brief resize_nearest_neighbor_v2 tiling data definition for kernel UT
 */

#ifndef RESIZE_NEAREST_NEIGHBOR_V2_TILING_DEFH
#define RESIZE_NEAREST_NEIGHBOR_V2_TILING_DEFH

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

// Define struct directly in UT directory (no dependency on BEGIN_TILING_DATA_DEF macro)
struct ResizeNearestNeighborV2TilingData {
    int64_t realCoreNum;
    int64_t ubSize;
    int64_t alignCorners;
    int64_t halfPixelCenters;
    int64_t lenN;
    int64_t lenC;
    int64_t lenSrcH;
    int64_t lenSrcW;
    int64_t lenDesH;
    int64_t lenDesW;
    int64_t condition;
    int64_t switchParams;
    int64_t splitBlockFactor;
    int64_t splitBlockTailFactor;
    int64_t lenCAlign;
    int64_t hwcNum;
    int64_t dstHwcNum;
    int64_t wcNum;
    int64_t dstWcNum;
    int64_t nLoop;
    int64_t nLoopTimesBefore;
    int64_t nLoopTimesLast;
    int64_t nLoopTailLast;
    int64_t wcLoop;
    int64_t wcLoopTimesBefore;
    int64_t wcLoopTailBefore;
    int64_t wcLoopTimesLast;
    int64_t wcLoopTailLast;
    int64_t splitBlockFullCount;
    int64_t splitFactorDesH;
    int64_t splitFactorTailDesH;
    int64_t splitCountDesH;
    int64_t splitFactorDesW;
    int64_t splitFactorTailDesW;
    int64_t splitCountDesW;
    float scaleW;
    float scaleH;
};

#define __aicore__
#ifdef __NPU_TILING__
inline __aicore__ void InitTilingData(const __gm__ uint8_t* tiling, ResizeNearestNeighborV2TilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(ResizeNearestNeighborV2TilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, ResizeNearestNeighborV2TilingData* constData)
{
    memcpy(constData, tiling, sizeof(ResizeNearestNeighborV2TilingData));
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

#define GET_TILING_DATA(tilingData, tilingArg)    \
    ResizeNearestNeighborV2TilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)

#define REGISTER_TILING_DEFAULT(T)

#endif // RESIZE_NEAREST_NEIGHBOR_V2_TILING_DEFH