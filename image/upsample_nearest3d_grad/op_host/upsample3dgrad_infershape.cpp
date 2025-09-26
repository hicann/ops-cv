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
 * \file upsample3dgrad_infershape.cpp
 * \brief
 */
#include <cmath>
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SUPPORTED_DIM_NUM = 5;
static constexpr size_t SUPPORTED_OUTPUT_DIM_NUM = 3;
static constexpr size_t INDEX_INPUT_SIZE = 0;
static constexpr size_t INDEX_OUTPUT_SIZE = 1;
static constexpr size_t INDEX_SCALES = 2;
static constexpr size_t NOT_CHANGE_DIM = 2;

static ge::graphStatus SetUpsample3dGradInferShape(const gert::InferShapeContext *context,
    const gert::Shape *grad_output_shape, gert::Shape *y_shape, const gert::ContinuousVector *input_size)
{
    auto attrs = context->GetAttrs();
    const gert::ContinuousVector *output_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_OUTPUT_SIZE);
    const gert::ContinuousVector *scales = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_SCALES);
    auto input_size_data = reinterpret_cast<const int64_t *>(input_size->GetData());
    auto output_size_data = reinterpret_cast<const int64_t *>(output_size->GetData());
    auto scales_data = reinterpret_cast<const float *>(scales->GetData());

    if (output_size->GetSize() != 0 && scales->GetSize() == 0) {
        OP_CHECK_IF(output_size->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            OP_LOGE(context->GetNodeName(), "attr::output_size dims must be 3, but get %zu", output_size->GetSize()),
            return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            OP_CHECK_IF(output_size_data[i] != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                OP_LOGE(context->GetNodeName(),
                    "attr::output_size[ %zu ](get %ld) != input::grad_output_size[ %zu ](get %ld).",
                    i,
                    output_size_data[i],
                    i + NOT_CHANGE_DIM,
                    grad_output_shape->GetDim(i + NOT_CHANGE_DIM)),
                return ge::GRAPH_FAILED);
        }
    } else if (output_size->GetSize() == 0 && scales->GetSize() != 0) {
        OP_CHECK_IF(scales->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            OP_LOGE(context->GetNodeName(), "attr::scales dims must be 3, but get %zu", scales->GetSize()),
            return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            int64_t tmp = int64_t(floor(input_size_data[i + NOT_CHANGE_DIM] * scales_data[i]));
            OP_CHECK_IF(tmp != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                OP_LOGE(context->GetNodeName(),
                    "input_size[ %zu ]*scales[ %zu ](get  %ld ) != grad_output_size[ %zu ](get  %ld ).", 
                    i + 2, i, tmp, i + NOT_CHANGE_DIM, grad_output_shape->GetDim(i + NOT_CHANGE_DIM)),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGE(context->GetNodeName(),
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
        return ge::GRAPH_FAILED;
    }

    *y_shape = *grad_output_shape;
    for (size_t i = 0; i < SUPPORTED_DIM_NUM; i++) {
        y_shape->SetDim(i, input_size_data[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Upsample3dGradInferShapeImpl(
    gert::InferShapeContext *context, const gert::Shape *grad_output_shape, gert::Shape *y_shape)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::ContinuousVector *input_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_INPUT_SIZE);
    OP_CHECK_IF(
        input_size == nullptr, OP_LOGE(context->GetNodeName(), "get attr::input_size faild!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(input_size->GetSize() != SUPPORTED_DIM_NUM,
        OP_LOGE(
            context->GetNodeName(), "attr::input_size dims must be 5, but get %zu", input_size->GetSize()),
        return ge::GRAPH_FAILED);

    return SetUpsample3dGradInferShape(context, grad_output_shape, y_shape, input_size);
}

static ge::graphStatus InferShape4Upsample3dGrad(gert::InferShapeContext *context)
{
    OP_LOGD(context, "begin to do InferShape4Upsample3dGrad");
    const gert::Shape *grad_output_shape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_output_shape);
    gert::Shape *y_shape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    auto grad_output_dim = grad_output_shape->GetDimNum();
    OP_CHECK_IF(grad_output_dim != SUPPORTED_DIM_NUM,
        OP_LOGE(
            context->GetNodeName(), "Expected dim of input x should be 5. but get %zu", grad_output_dim),
        return ge::GRAPH_FAILED);

    return Upsample3dGradInferShapeImpl(context, grad_output_shape, y_shape);
}

IMPL_OP_INFERSHAPE(UpsampleNearest3dGrad).InferShape(InferShape4Upsample3dGrad);

}  // namespace ops
