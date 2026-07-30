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
 * \file grid_unnormal_infershape.cpp
 * \brief GridUnnormal 形状/类型推导：
 *   diff/position 的 shape 与 grid 完全一致；
 *   diff dtype = grid dtype，position dtype = int32。
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <string>

using namespace ge;

namespace ops {

static constexpr size_t kInputGridIdx = 0;
static constexpr size_t kInputAssistIdx = 1;
static constexpr size_t kOutputDiffIdx = 0;
static constexpr size_t kOutputPosIdx = 1;
static constexpr size_t kGridRank = 4;
static constexpr size_t kLastDimIdx = 3;
static constexpr int64_t kCoordDim = 2;
static constexpr int64_t kUnknownRankDim = -2;

static bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == kUnknownRankDim;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext* context, const gert::Shape* gridShape,
                                       const gert::Shape* assistShape)
{
    if (IsUnknownRank(gridShape)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        gridShape->GetDimNum() != kGridRank,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "rank is not 4", "grid rank must be 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape->GetDim(kLastDimIdx) != ge::UNKNOWN_DIM && gridShape->GetDim(kLastDimIdx) != kCoordDim,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "last dim is not 2",
                                                       "grid last dim must be 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(assistShape->GetDimNum() != gridShape->GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "assist", "rank is not equal to grid",
                                                       "grid and assist rank must be equal"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < gridShape->GetDimNum(); ++i) {
        const int64_t gridDim = gridShape->GetDim(i);
        const int64_t assistDim = assistShape->GetDim(i);
        OP_CHECK_IF(gridDim != ge::UNKNOWN_DIM && assistDim != ge::UNKNOWN_DIM && gridDim != assistDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid and assist", "not equal",
                                                           "known dims of grid and assist must be equal"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GridUnnormal(gert::InferShapeContext* context)
{
    const gert::Shape* gridShape = context->GetInputShape(kInputGridIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    const gert::Shape* assistShape = context->GetInputShape(kInputAssistIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistShape);
    if (CheckInputShape(context, gridShape, assistShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* diffShape = context->GetOutputShape(kOutputDiffIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffShape);
    gert::Shape* posShape = context->GetOutputShape(kOutputPosIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, posShape);

    *diffShape = *gridShape;
    *posShape = *gridShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GridUnnormal(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(
        context->SetOutputDataType(kOutputDiffIdx, context->GetInputDataType(kInputGridIdx)) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "diff",
                                              std::to_string(context->GetInputDataType(kInputGridIdx)).c_str(),
                                              "SetOutputDataType failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(kOutputPosIdx, ge::DT_INT32) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "position",
                                                      std::to_string(ge::DT_INT32).c_str(), "SetOutputDataType failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridUnnormal).InferShape(InferShape4GridUnnormal).InferDataType(InferDtype4GridUnnormal);

} // namespace ops
