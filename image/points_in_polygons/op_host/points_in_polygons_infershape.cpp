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
 * \file points_in_polygons_infershape.cpp
 * \brief PointsInPolygons shape inference: output.shape = (N, M)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4PointsInPolygons(gert::InferShapeContext* context)
{
    const gert::Shape* pointsShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pointsShape);
    const gert::Shape* polygonsShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, polygonsShape);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // output.shape = (points.shape[0], polygons.shape[1]) = (N, M)
    int64_t N = (pointsShape->GetDimNum() >= 1) ? pointsShape->GetDim(0) : 0;
    int64_t M = (polygonsShape->GetDimNum() >= 2) ? polygonsShape->GetDim(1) : 0;
    outputShape->SetDimNum(0);
    outputShape->AppendDim(N);
    outputShape->AppendDim(M);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PointsInPolygons).InferShape(InferShape4PointsInPolygons);

} // namespace ops
