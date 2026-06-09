/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file c_io_u_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeCIoU(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCIoU");

    const gert::Shape* bboxesShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, bboxesShape);

    int64_t n = bboxesShape->GetDimNum() >= 2 ? bboxesShape->GetDim(1) : 0;

    gert::Shape* overlapShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, overlapShape);
    overlapShape->SetDimNum(2);
    overlapShape->SetDim(0, 1);
    overlapShape->SetDim(1, n);

    gert::Shape* atanSubShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, atanSubShape);
    atanSubShape->SetDimNum(2);
    atanSubShape->SetDim(0, 1);
    atanSubShape->SetDim(1, n);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCIoU");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCIoU(gert::InferDataTypeContext* context)
{
    const auto inDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inDtype);
    context->SetOutputDataType(1, inDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CIoU).InferShape(InferShapeCIoU).InferDataType(InferDataTypeCIoU);
} // namespace ops
