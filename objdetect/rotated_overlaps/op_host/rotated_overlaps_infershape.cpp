/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace {
constexpr int64_t kBoxesIndex = 0;
constexpr int64_t kQueryBoxesIndex = 1;
constexpr int64_t kOutputIndex = 0;
constexpr int64_t kCoordinateCount = 5;
constexpr int64_t kMaxQueries = 2000;

bool IsKnown(int64_t dim) { return dim != ge::UNKNOWN_DIM; }

bool IsKnownPositive(int64_t dim) { return !IsKnown(dim) || dim > 0; }

bool ProductFitsInt64(int64_t a, int64_t b, int64_t c)
{
    if (!IsKnown(a) || !IsKnown(b) || !IsKnown(c)) {
        return true;
    }
    if (a <= 0 || b <= 0 || c <= 0) {
        return false;
    }
    const uint64_t ua = static_cast<uint64_t>(a);
    const uint64_t ub = static_cast<uint64_t>(b);
    const uint64_t uc = static_cast<uint64_t>(c);
    return ua <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / ub &&
           ua * ub <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / uc;
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForRotatedOverlaps(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(kBoxesIndex);
    const gert::Shape* queryBoxesShape = context->GetInputShape(kQueryBoxesIndex);
    gert::Shape* outputShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryBoxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    if (boxesShape->GetDimNum() == ge::UNKNOWN_RANK.size() || queryBoxesShape->GetDimNum() == ge::UNKNOWN_RANK.size()) {
        outputShape->SetDimNum(ge::UNKNOWN_RANK.size());
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(boxesShape->GetDimNum() != 3 || queryBoxesShape->GetDimNum() != 3,
                OP_LOGE(context, "RotatedOverlaps inputs must both be rank 3."), return ge::GRAPH_FAILED);

    const int64_t batch = boxesShape->GetDim(0);
    const int64_t queryBatch = queryBoxesShape->GetDim(0);
    const int64_t numBoxes = boxesShape->GetDim(2);
    const int64_t numQueries = queryBoxesShape->GetDim(2);
    OP_CHECK_IF((IsKnown(boxesShape->GetDim(1)) && boxesShape->GetDim(1) != kCoordinateCount) ||
                    (IsKnown(queryBoxesShape->GetDim(1)) && queryBoxesShape->GetDim(1) != kCoordinateCount),
                OP_LOGE(context, "RotatedOverlaps channel dimension must be 5."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsKnownPositive(batch) || !IsKnownPositive(queryBatch) || !IsKnownPositive(numBoxes) ||
                    !IsKnownPositive(numQueries),
                OP_LOGE(context, "RotatedOverlaps known dimensions B, N and K must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        IsKnown(numQueries) && numQueries > kMaxQueries,
        OP_LOGE(context, "RotatedOverlaps supports at most %ld query boxes, but got %ld.", kMaxQueries, numQueries),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsKnown(batch) && IsKnown(queryBatch) && batch != queryBatch,
                OP_LOGE(context, "RotatedOverlaps batch dimensions must match."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ProductFitsInt64(batch, numBoxes, numQueries),
                OP_LOGE(context, "RotatedOverlaps output element count overflows int64."), return ge::GRAPH_FAILED);

    outputShape->SetDimNum(3);
    outputShape->SetDim(0, batch);
    outputShape->SetDim(1, numBoxes);
    outputShape->SetDim(2, numQueries);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRotatedOverlaps(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutputIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotatedOverlaps)
    .InferShape(InferShapeForRotatedOverlaps)
    .InferDataType(InferDataTypeForRotatedOverlaps);
} // namespace ops
