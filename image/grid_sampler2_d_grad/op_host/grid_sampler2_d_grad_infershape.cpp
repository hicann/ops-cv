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
 * \file grid_sampler2_d_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

static constexpr int INPUT_GRAD_INDEX = 0;
static constexpr int INPUT_X_INDEX = 1;
static constexpr int INPUT_GRID_INDEX = 2;
static constexpr int OUTPUT_DX_INDEX = 0;
static constexpr int OUTPUT_DGRID_INDEX = 1;
static constexpr int GRID_SAMPLER2D_GRAD_SHAPE_LIMIT = 4;
using namespace ge;
using namespace Ops::Base;
namespace ops {
static ge::graphStatus InferShape4GridSampler2DGrad(gert::InferShapeContext* context)
{
    // infer shape
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("GridSampler2DGrad", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GridSampler2DGrad.");

    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* gridShape = context->GetInputShape(INPUT_GRID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);

    OP_CHECK_IF(
        (xShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT ||
         gridShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT ||
         gradShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT),
        OP_LOGE(context->GetNodeName(), "shape is invalid, only support rank is 4."), return ge::GRAPH_FAILED);

    gert::Shape* outDxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDxShape);

    gert::Shape* outDgridShape = context->GetOutputShape(OUTPUT_DGRID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDgridShape);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*outDxShape);
    } else {
        *outDxShape = *xShape;
    }

    if (IsUnknownRank(*gridShape)) {
        SetUnknownRank(*outDgridShape);
    } else {
        *outDgridShape = *gridShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4GridSampler2DGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4GridSampler2DGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GridSampler2DGrad");
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_DGRID_INDEX, context->GetInputDataType(INPUT_GRID_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4GridSampler2DGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSampler2DGrad)
    .InferShape(InferShape4GridSampler2DGrad)
    .InferDataType(InferDataType4GridSampler2DGrad);
} // namespace ops