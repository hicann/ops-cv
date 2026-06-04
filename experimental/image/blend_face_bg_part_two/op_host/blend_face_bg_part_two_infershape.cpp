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
 * \file blend_face_bg_part_two_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeBlendFaceBgPartTwo(gert::InferShapeContext* context)
{
    const gert::Shape* accFaceShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, accFaceShape);
    gert::Shape* fusedImgShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, fusedImgShape);

    *fusedImgShape = *accFaceShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlendFaceBgPartTwo(gert::InferDataTypeContext* context)
{
    const auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlendFaceBgPartTwo)
    .InferShape(InferShapeBlendFaceBgPartTwo)
    .InferDataType(InferDataTypeBlendFaceBgPartTwo);
} // namespace ops
