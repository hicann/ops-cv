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
 * \file three_interpolate_backward.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace {
const uint32_t INDEX_INPUT_GRAD_X = 0u;
const uint32_t INDEX_OUTPUT_GRAD_Y = 0u;
const int32_t UNKNOW_DIM = -1;
enum DIM { DIM_0, DIM_1, DIM_2, DIM_3, DIM_4, DIM_5 };
} // namespace

namespace ops {
static graphStatus InferShape4ThreeInterpolateBackward(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter InferShapeThreeInterpolateBackward");
    OP_LOGI(context, "Enter InferShape4ThreeInterpolateBackward");

    const gert::Shape* grad_x_shape = context->GetInputShape(INDEX_INPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_x_shape);

    gert::Shape* grad_y_shape = context->GetOutputShape(INDEX_OUTPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_y_shape);

    if (Ops::Base::IsUnknownRank(*grad_x_shape)) {
        OP_LOGI(context, "input is UnknownRank, set output as UnknownRank.");
        Ops::Base::SetUnknownRank(*grad_y_shape);
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto attr_pointer = attrs->GetAttrPointer<uint32_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_pointer);
    auto ms = *attr_pointer;
    // -1 维直接透传，避免 uint32_t 截断为巨大正值
    int64_t bs = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_0));
    int64_t c1 = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_1));
    int64_t c0 = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_4));

    grad_y_shape->SetDimNum(DIM_5);
    grad_y_shape->SetDim(DIM_0, bs);
    grad_y_shape->SetDim(DIM_1, c1);
    grad_y_shape->SetDim(DIM_2, ms);
    grad_y_shape->SetDim(DIM_3, 1);
    grad_y_shape->SetDim(DIM_4, c0);

    OP_LOGI(
        context, "Infershape N:%ld C1:%ld H:%ld W:%ld C0:%ld.", grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_0)),
        grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_1)), grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_2)),
        grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_3)), grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_4)));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ThreeInterpolateBackward(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ThreeInterpolateBackward");
    const ge::DataType input_grad_x_dtype = context->GetInputDataType(INDEX_INPUT_GRAD_X);
    context->SetOutputDataType(INDEX_OUTPUT_GRAD_Y, input_grad_x_dtype);
    OP_LOGD(context, "End to do InferDataType4ThreeInterpolateBackward");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ThreeInterpolateBackward)
    .InferShape(InferShape4ThreeInterpolateBackward)
    .InferDataType(InferDataType4ThreeInterpolateBackward);
} // namespace ops
