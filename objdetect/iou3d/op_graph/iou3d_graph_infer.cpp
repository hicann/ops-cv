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
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// dtype 推导：两个输入都必须为 float32，输出固定为 float32。
static ge::graphStatus InferDataType4Iou3D(gert::InferDataTypeContext* context)
{
    const auto bboxesDtype = context->GetInputDataType(0);
    const auto gtboxesDtype = context->GetInputDataType(1);
    OP_CHECK_IF(bboxesDtype != ge::DT_FLOAT || gtboxesDtype != ge::DT_FLOAT,
                OP_LOGE(context, "Iou3D: bboxes/gtboxes must both be float32, got %d/%d", static_cast<int>(bboxesDtype),
                        static_cast<int>(gtboxesDtype)),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Iou3D).InferDataType(InferDataType4Iou3D);

} // namespace ops
