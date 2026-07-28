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
 * \file iou3d_tiling.h
 * \brief Iou3D kernel UT 的 tiling shim（CPU 仿真侧）。
 *
 * AddOpTestCase 宏会自动 `-include` 本文件（若存在），用于在 tikicpulib CPU 仿真下替换
 * kernel 里的 REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT 宏：真机侧这两个宏
 * 从 __gm__ 读 tiling 并做寄存器化，CPU 仿真侧改为 memcpy 到栈上结构体。
 * 结构体定义复用算子真实 op_kernel/arch35/iou3d_tiling_data.h（单一真值源，不重定义字段）。
 */

#ifndef IOU3D_KERNEL_UT_TILING_H
#define IOU3D_KERNEL_UT_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/arch35/iou3d_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, Iou3DTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(Iou3DTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, Iou3DTilingData* constData)
{
    memcpy(constData, tiling, sizeof(Iou3DTilingData));
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
    Iou3DTilingData tilingData;                \
    InitTilingData(tilingArg, &tilingData)

#define REGISTER_TILING_DEFAULT(T)

#endif // IOU3D_KERNEL_UT_TILING_H
