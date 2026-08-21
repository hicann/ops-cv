/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

#include <cstdint>
#include <vector>

#include "graph/types.h"

using namespace ge;

namespace ops {

namespace {

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

} // namespace

/**
 * InferShapeForCheckValid: GE shape inference callback.
 *
 * valid_tensor.shape = (bbox_tensor.shape[0], 1) = (N, 1).
 * Validates that bbox_tensor is rank-2 with last dim == 4.
 * Empty tensor (N==0) is supported 鈥?output (0, 1) propagates.
 */
ge::graphStatus InferShapeForCheckValid(gert::InferShapeContext* context)
{
    const gert::Shape* bboxShape = context->GetInputShape(0);
    gert::Shape* validShape = context->GetOutputShape(0);
    if (bboxShape == nullptr || validShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsUnknownRank(bboxShape)) {
        validShape->SetDimNum(1);
        validShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    const size_t rank = bboxShape->GetDimNum();
    if (rank != 2) {
        return GRAPH_FAILED;
    }
    if (bboxShape->GetDim(1) != 4) {
        return GRAPH_FAILED;
    }

    validShape->SetDimNum(2);
    validShape->SetDim(0, bboxShape->GetDim(0));
    validShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CheckValid).InferShape(InferShapeForCheckValid);

} // namespace ops
