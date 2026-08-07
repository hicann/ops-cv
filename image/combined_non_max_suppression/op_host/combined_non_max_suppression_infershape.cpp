/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int32_t BOXES_INDEX = 0;
constexpr int32_t SCORES_INDEX = 1;
constexpr int32_t MAX_PER_CLASS_INDEX = 2;
constexpr int32_t MAX_TOTAL_INDEX = 3;
constexpr int32_t ATTR_PAD_PER_CLASS_INDEX = 0;
constexpr int32_t BOXES_RANK = 4;
constexpr int32_t SCORES_RANK = 3;
constexpr int32_t BOX_COORDS = 4;
constexpr int32_t OUTPUT_BOXES_RANK = 3;
constexpr int32_t OUTPUT_VECTOR_RANK = 2;
constexpr int32_t OUTPUT_VALID_RANK = 1;
constexpr int32_t MAX_OUTPUT_SIZE = 1000;
constexpr int32_t MAX_CLASSES = 200;
constexpr int32_t MAX_NUM_BOXES = 200000;

template <typename T>
ge::graphStatus ReadScalar(gert::InferShapeContext* context, int32_t index, T& value)
{
    const gert::Tensor* tensor = context->GetInputTensor(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "input %d scalar data is null", index), return ge::GRAPH_FAILED);
    value = data[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCombinedNonMaxSuppression(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(BOXES_INDEX);
    const gert::Shape* scoresShape = context->GetInputShape(SCORES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresShape);
    OP_CHECK_IF(boxesShape->GetDimNum() != BOXES_RANK,
                OP_LOGE(context, "boxes must be 4D, got %zuD", boxesShape->GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(scoresShape->GetDimNum() != SCORES_RANK,
                OP_LOGE(context, "scores must be 3D, got %zuD", scoresShape->GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t batch = boxesShape->GetDim(0);
    const int64_t numBoxes = boxesShape->GetDim(1);
    const int64_t boxClasses = boxesShape->GetDim(2);
    const int64_t classes = scoresShape->GetDim(2);
    OP_CHECK_IF(batch <= 0 || numBoxes <= 0 || classes <= 0,
                OP_LOGE(context, "boxes and scores dimensions must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numBoxes > MAX_NUM_BOXES || classes > MAX_CLASSES,
                OP_LOGE(context, "num_boxes must be <= %d and num_classes must be <= %d", MAX_NUM_BOXES, MAX_CLASSES),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesShape->GetDim(3) != BOX_COORDS, OP_LOGE(context, "boxes last dimension must be 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scoresShape->GetDim(0) != batch || scoresShape->GetDim(1) != numBoxes,
                OP_LOGE(context, "boxes and scores batch/num_boxes dimensions must match"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxClasses != 1 && boxClasses != classes,
                OP_LOGE(context, "boxes q dimension must be 1 or equal to num_classes"), return ge::GRAPH_FAILED);

    int32_t maxPerClass = 0;
    int32_t maxTotal = 0;
    OP_CHECK_IF(ReadScalar(context, MAX_PER_CLASS_INDEX, maxPerClass) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read max_output_size_per_class"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, MAX_TOTAL_INDEX, maxTotal) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read max_total_size"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxPerClass <= 0 || maxPerClass > MAX_OUTPUT_SIZE || maxTotal <= 0 || maxTotal > MAX_OUTPUT_SIZE,
                OP_LOGE(context, "max output sizes must be in [1, %d]", MAX_OUTPUT_SIZE), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* padPerClass = attrs->GetBool(ATTR_PAD_PER_CLASS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padPerClass);
    int64_t outputSize = maxTotal;
    if (*padPerClass) {
        outputSize = std::min<int64_t>(maxTotal, static_cast<int64_t>(maxPerClass) * classes);
    }

    gert::Shape* outBoxes = context->GetOutputShape(0);
    gert::Shape* outScores = context->GetOutputShape(1);
    gert::Shape* outClasses = context->GetOutputShape(2);
    gert::Shape* outValid = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, outBoxes);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScores);
    OP_CHECK_NULL_WITH_CONTEXT(context, outClasses);
    OP_CHECK_NULL_WITH_CONTEXT(context, outValid);
    outBoxes->SetDimNum(OUTPUT_BOXES_RANK);
    outBoxes->SetDim(0, batch);
    outBoxes->SetDim(1, outputSize);
    outBoxes->SetDim(2, BOX_COORDS);
    outScores->SetDimNum(OUTPUT_VECTOR_RANK);
    outScores->SetDim(0, batch);
    outScores->SetDim(1, outputSize);
    *outClasses = *outScores;
    outValid->SetDimNum(OUTPUT_VALID_RANK);
    outValid->SetDim(0, batch);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCombinedNonMaxSuppression(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(CombinedNonMaxSuppression)
    .InferShape(InferShapeCombinedNonMaxSuppression)
    .InferDataType(InferDataTypeCombinedNonMaxSuppression)
    .InputsDataDependency({MAX_PER_CLASS_INDEX, MAX_TOTAL_INDEX});

} // namespace ops
