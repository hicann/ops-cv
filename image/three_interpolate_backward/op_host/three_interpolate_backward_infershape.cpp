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
enum DIM
{
    DIM_0,
    DIM_1,
    DIM_2,
    DIM_3,
    DIM_4,
    DIM_5
};
} // namespace

namespace ops {
static graphStatus InferShape4ThreeInterpolateBackward(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Enter InferShape4ThreeInterpolateBackward");

    const gert::Shape* grad_x_shape = context->GetInputShape(INDEX_INPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_x_shape);

    gert::Shape* grad_y_shape = context->GetOutputShape(INDEX_OUTPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_y_shape);

    if (Ops::Base::IsUnknownRank(*grad_x_shape)) {
        OP_LOGI(context, "grad_x_shape is unknow rank set output all -1");
        grad_y_shape->SetDimNum(DIM_5);
        grad_y_shape->SetDim(DIM_0, UNKNOW_DIM);
        grad_y_shape->SetDim(DIM_1, UNKNOW_DIM);
        grad_y_shape->SetDim(DIM_2, UNKNOW_DIM);
        grad_y_shape->SetDim(DIM_3, UNKNOW_DIM);
        grad_y_shape->SetDim(DIM_4, UNKNOW_DIM);
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto attr_pointer = attrs->GetAttrPointer<uint32_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_pointer);
    auto ms = *attr_pointer;
    uint32_t bs = grad_x_shape->GetDim(DIM_0);
    uint32_t c1 = grad_x_shape->GetDim(DIM_1);
    uint32_t c0 = grad_x_shape->GetDim(DIM_4);

    grad_y_shape->SetDimNum(DIM_5);
    grad_y_shape->SetDim(DIM_0, bs);
    grad_y_shape->SetDim(DIM_1, c1);
    grad_y_shape->SetDim(DIM_2, ms);
    grad_y_shape->SetDim(DIM_3, 1);
    grad_y_shape->SetDim(DIM_4, c0);

    OP_LOGI(
        context, "Intershape N:%ld C1:%ld H:%ld W:%ld C0:%ld.", grad_y_shape->GetDim(DIM_0),
        grad_y_shape->GetDim(DIM_1), grad_y_shape->GetDim(DIM_2), grad_y_shape->GetDim(DIM_3),
        grad_y_shape->GetDim(DIM_4));

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