/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aipp_tiling.h
 * \brief aipp tiling data definition for kernel UT
 */

#ifndef AIPP_TILING_DEFH
#define AIPP_TILING_DEFH

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/arch35/aipp_struct.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace Aipp_Kernel;

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, AippTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (auto i = 0; i < sizeof(AippTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, AippTilingData *constData)
{
    memcpy(constData, tiling, sizeof(AippTilingData));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    AippTilingData tilingData;                 \
    InitTilingData(tilingArg, &tilingData)

#define REGISTER_TILING_DEFAULT(T)

#endif // AIPP_TILING_DEFH