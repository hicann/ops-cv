/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace {
constexpr int64_t kBoxesIndex = 0;
constexpr int64_t kScoresIndex = 1;
constexpr int64_t kNmsedBoxesIndex = 0;
constexpr int64_t kNmsedScoresIndex = 1;
constexpr int64_t kNmsedClassesIndex = 2;
constexpr int64_t kNmsedNumIndex = 3;
constexpr int64_t kMaxTotalSizeAttrIndex = 3;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForBatchMultiClassNonMaxSuppression(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(kBoxesIndex);
    const gert::Shape* scoresShape = context->GetInputShape(kScoresIndex);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    OP_CHECK_IF(boxesShape->GetDimNum() != 4 || scoresShape->GetDimNum() != 3,
                OP_LOGE(context, "boxes must be rank 4 and scores must be rank 3."), return ge::GRAPH_FAILED);
    const int64_t batch = boxesShape->GetDim(0);
    const int64_t scoresBatch = scoresShape->GetDim(0);
    const int64_t* maxTotalSize = attrs->GetAttrPointer<int64_t>(kMaxTotalSizeAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxTotalSize);
    OP_CHECK_IF(batch <= 0 || scoresBatch != batch || *maxTotalSize <= 0,
                OP_LOGE(context, "Invalid batch dimension or max_total_size."), return ge::GRAPH_FAILED);

    gert::Shape* nmsedBoxesShape = context->GetOutputShape(kNmsedBoxesIndex);
    gert::Shape* nmsedScoresShape = context->GetOutputShape(kNmsedScoresIndex);
    gert::Shape* nmsedClassesShape = context->GetOutputShape(kNmsedClassesIndex);
    gert::Shape* nmsedNumShape = context->GetOutputShape(kNmsedNumIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedBoxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedScoresShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedClassesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedNumShape);

    nmsedBoxesShape->SetDimNum(3);
    nmsedBoxesShape->SetDim(0, batch);
    nmsedBoxesShape->SetDim(1, *maxTotalSize);
    nmsedBoxesShape->SetDim(2, 4);
    for (gert::Shape* output : {nmsedScoresShape, nmsedClassesShape}) {
        output->SetDimNum(2);
        output->SetDim(0, batch);
        output->SetDim(1, *maxTotalSize);
    }
    nmsedNumShape->SetDimNum(1);
    nmsedNumShape->SetDim(0, batch);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForBatchMultiClassNonMaxSuppression(gert::InferDataTypeContext* context)
{
    const ge::DataType boxesType = context->GetInputDataType(kBoxesIndex);
    context->SetOutputDataType(kNmsedBoxesIndex, boxesType);
    context->SetOutputDataType(kNmsedScoresIndex, boxesType);
    context->SetOutputDataType(kNmsedClassesIndex, boxesType);
    context->SetOutputDataType(kNmsedNumIndex, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchMultiClassNonMaxSuppression)
    .InferShape(InferShapeForBatchMultiClassNonMaxSuppression)
    .InferDataType(InferDataTypeForBatchMultiClassNonMaxSuppression);
} // namespace ops
