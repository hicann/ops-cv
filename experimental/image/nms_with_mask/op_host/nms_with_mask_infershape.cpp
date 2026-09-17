/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file nms_with_mask_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_X_INDEX = 0U;
static constexpr size_t OUTPUT_Y_INDEX = 0U;
static constexpr size_t INPUT_RANK = 2U;
static constexpr size_t OUTPUT_RANK = 1U;
static constexpr size_t COORDINATE_DIM_INDEX = 1U;
static constexpr int64_t COORDINATE_DIM = 4;

static ge::graphStatus InferShapeNMSWithMask(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeNMSWithMask");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    const size_t xShapeSize = xShape->GetDimNum();
    OP_CHECK_IF(xShapeSize != INPUT_RANK,
                OP_LOGE(context, "NMSWithMask input x must be rank %zu, got %zu", INPUT_RANK, xShapeSize),
                return GRAPH_FAILED);

    const int64_t coordinateDim = xShape->GetDim(COORDINATE_DIM_INDEX);
    OP_CHECK_IF(
        coordinateDim != ge::UNKNOWN_DIM && coordinateDim != COORDINATE_DIM,
        OP_LOGE(context, "NMSWithMask input x second dimension must be %ld, got %ld", COORDINATE_DIM, coordinateDim),
        return GRAPH_FAILED);

    yShape->SetDimNum(OUTPUT_RANK);
    yShape->SetDim(0U, xShape->GetDim(0U));

    OP_LOGD(context->GetNodeName(), "End to do InferShapeNMSWithMask");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NMSWithMask).InferShape(InferShapeNMSWithMask);
} // namespace ops
