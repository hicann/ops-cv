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
 * \file resize_linear_grad_infershape.cpp
 * \brief resize_linear_grad_infershape
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_GRADS = 0;
static constexpr size_t IN_ORIGINAL_IMAGE = 1;
static constexpr size_t OUT_Y = 0;
static constexpr size_t IN_GRADS_DIMS = 3;
static constexpr size_t IN_SIZE_NUM = 1;
static constexpr size_t IDX_L = 2;

ge::graphStatus ResizeLinearGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeLinearGradInferShape rt2.0");

    auto nodeName = context->GetNodeName();

    auto yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto imageShape = context->GetInputShape(IN_ORIGINAL_IMAGE);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageShape);

    if (Ops::Base::IsUnknownRank(*imageShape)) {
        Ops::Base::SetUnknownShape(IN_GRADS_DIMS, *yShape);
    } else {
        OP_CHECK_IF(
            imageShape->GetDimNum() != IN_GRADS_DIMS, OP_LOGE(nodeName, "original_image shape only support 3D"),
            return GRAPH_FAILED);
        OP_CHECK_IF(
            (imageShape->GetDim(IDX_L) == 0) || (imageShape->GetDim(IDX_L - 1) == 0),
            OP_LOGE(nodeName, "original_image size should be greater than 0"), return GRAPH_FAILED);
        *yShape = *imageShape;
    }

    OP_LOGI(context->GetNodeName(), "End to do ResizeLinearGradInferShape rt2.0");
    return ge::GRAPH_SUCCESS;
}

graphStatus ResizeLinearGradInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeLinearGradInferDtype");

    auto gradsDtype = context->GetInputDataType(IN_GRADS);
    OP_CHECK_IF(
        (gradsDtype != ge::DT_FLOAT) && (gradsDtype != ge::DT_FLOAT16) && (gradsDtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "grads dtype only support float, float16 and bfloat16"), return GRAPH_FAILED);

    context->SetOutputDataType(OUT_Y, gradsDtype);

    OP_LOGI(context->GetNodeName(), "End to do ResizeLinearGradInferDtype");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeLinearGrad).InferShape(ResizeLinearGradInferShape).InferDataType(ResizeLinearGradInferDtype);
} // namespace ops
