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
 * \file upsample_nearest_exact3d_tiling.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST_EXACT3D_TILING_DEFH
#define UPSAMPLE_NEAREST_EXACT3D_TILING_DEFH

#include <cstdint>

#include "../../../../upsample_nearest3d/op_kernel/upsample_nearest3d_struct.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace UpsampleNearest3d;

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, UpsampleNearest3dTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (auto i = 0; i < sizeof(UpsampleNearest3dTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, UpsampleNearest3dTilingData *constData)
{
    memcpy(constData, tiling, sizeof(UpsampleNearest3dTilingData));
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
    UpsampleNearest3dTilingData tilingData;    \
    InitTilingData(tilingArg, &tilingData)

#endif // UPSAMPLE_NEAREST_EXACT3D_TILING_DEFH