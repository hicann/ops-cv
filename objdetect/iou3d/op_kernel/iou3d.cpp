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
 * \file iou3d.cpp
 * \brief Iou3D 算子 kernel 入口（arch35 / DAV_3510）
 *
 * 一芯片一算子一入口：arch35 唯一 __global__ 入口。
 * 无 TilingKey 参数，单 dtype float32（由 def 文件驱动）。
 *   - 空 Tensor 短路通过运行时判断（Init/Process 中判断 isEmpty_）
 *
 * def 驱动 dtype：dtype 由 _def.cpp 的 DataType({ge::DT_FLOAT}) 声明，构建系统通过
 * -DDTYPE_BBOXES 等编译宏注入，kernel 直接使用 DTYPE_BBOXES 宏获取实际类型。
 */

#include "arch35/iou3d.h"

// 核函数入口名须为 OpType(Iou3D) 的 snake_case 形式：iou3_d（框架 CamelCase→snake_case 约定）。
__global__ __aicore__ void iou3_d(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR iou, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Iou3DTilingData);
    GET_TILING_DATA_WITH_STRUCT(Iou3DTilingData, tilingData, tiling);
    NsIou3D::Iou3D op;
    op.Init(bboxes, gtboxes, iou, &tilingData);
    op.Process();
}
