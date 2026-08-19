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
 * InferShapeForBoundingBoxDecode: GE shape inference callback.
 *
 * boxes.shape = anchor_box.shape (identity).  Validates that both inputs are
 * non-null, rank == 2, last dim == 4, and the two input shapes are equal
 * (DESIGN §3.1).  Empty tensor (N==0) is supported — the identity copy
 * propagates (0, 4) to the output.
 */
ge::graphStatus InferShapeForBoundingBoxDecode(gert::InferShapeContext* context)
{
    const gert::Shape* anchorShape = context->GetInputShape(0);
    const gert::Shape* deltasShape = context->GetInputShape(1);
    gert::Shape* boxesShape = context->GetOutputShape(0);
    if (anchorShape == nullptr || deltasShape == nullptr || boxesShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsUnknownRank(anchorShape) || IsUnknownRank(deltasShape)) {
        boxesShape->SetDimNum(1);
        boxesShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    const size_t rank = anchorShape->GetDimNum();
    if (rank != 2 || deltasShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (anchorShape->GetDim(1) != 4 || deltasShape->GetDim(1) != 4) {
        return GRAPH_FAILED;
    }

    // Validate anchor_box.shape == deltas.shape (no broadcast, DESIGN §3.1)
    const int64_t anchorN = anchorShape->GetDim(0);
    const int64_t deltasN = deltasShape->GetDim(0);
    if (anchorN != deltasN && anchorN != ge::UNKNOWN_DIM && deltasN != ge::UNKNOWN_DIM) {
        return GRAPH_FAILED;
    }

    // boxes.shape = anchor_box.shape (identity copy)
    boxesShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        boxesShape->SetDim(i, anchorShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

/**
 * IMPL_OP_INFERSHAPE(BoundingBoxDecode).InferShape(...):
 *   Registers the shape inference function at static init time.  When the
 *   framework needs to determine the output shape of a BoundingBoxDecode node,
 *   it calls InferShapeForBoundingBoxDecode.  This static registration is
 *   required by the GE runtime (NnopbaseExecutorDoTiling) to locate the op
 *   implementation via the op_impl_registry.
 */
IMPL_OP_INFERSHAPE(BoundingBoxDecode).InferShape(InferShapeForBoundingBoxDecode);

} // namespace ops
