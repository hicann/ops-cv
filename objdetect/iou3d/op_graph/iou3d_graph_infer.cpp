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
 * \file iou3d_graph_infer.cpp
 * \brief Iou3D 算子图模式数据类型推导实现
 *
 * dtype_rule:
 *   iou.dtype = bboxes.dtype == float32
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

// dtype 推导：iou.dtype = bboxes.dtype
static ge::graphStatus InferDataType4Iou3D(gert::InferDataTypeContext* context)
{
    const auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Iou3D).InferDataType(InferDataType4Iou3D);

} // namespace ops
