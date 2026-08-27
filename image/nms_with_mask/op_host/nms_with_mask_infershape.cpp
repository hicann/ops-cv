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
 * \file nms_with_mask_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
// ---------------- NMSWithMask Op-------------------
constexpr size_t INPUT_DIM_NUM = 2;
constexpr int64_t BOX_SCORES_DIM_NUM = 5;
constexpr size_t INPUT_BOX_SCORES_INDEX = 0;
constexpr size_t BOX_SCORES_DIM_INDEX = 1;
constexpr size_t BOXES_NUM_DIM_INDEX = 0;
constexpr size_t SELECTED_BOXES_OUTPUT_INDEX = 0;
constexpr size_t VECTOR_OUTPUT_DIM_NUM = 1;
constexpr size_t OUTPUT_TENSOR_NUM = 3;

static graphStatus InferShape4NMSWithMask(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do NMSWithMaskInferShape");
    const gert::Shape* input_scores_shape = context->GetInputShape(INPUT_BOX_SCORES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_scores_shape);
    if (!Ops::Base::IsUnknownRank(*input_scores_shape)) {
        OP_CHECK_IF(input_scores_shape->GetDimNum() != INPUT_DIM_NUM,
                    OP_LOGE(context, "Input box_scores shape only supports %zu-D, got %zu-D.", INPUT_DIM_NUM,
                            input_scores_shape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(input_scores_shape->GetDim(BOX_SCORES_DIM_INDEX) != ge::UNKNOWN_DIM &&
                        input_scores_shape->GetDim(BOX_SCORES_DIM_INDEX) != BOX_SCORES_DIM_NUM,
                    OP_LOGE(context, "Input box_scores second dim must be %ld, got %ld.", BOX_SCORES_DIM_NUM,
                            input_scores_shape->GetDim(BOX_SCORES_DIM_INDEX)),
                    return ge::GRAPH_FAILED);
    }

    for (size_t i = 0; i < OUTPUT_TENSOR_NUM; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
        if (i == SELECTED_BOXES_OUTPUT_INDEX) {
            output_shape->SetDimNum(INPUT_DIM_NUM);
            output_shape->SetDim(BOXES_NUM_DIM_INDEX, input_scores_shape->GetDim(BOXES_NUM_DIM_INDEX));
            output_shape->SetDim(BOX_SCORES_DIM_INDEX, BOX_SCORES_DIM_NUM);
        } else {
            output_shape->SetDimNum(VECTOR_OUTPUT_DIM_NUM);
            output_shape->SetDim(BOXES_NUM_DIM_INDEX, input_scores_shape->GetDim(BOXES_NUM_DIM_INDEX));
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do NMSWithMaskInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NMSWithMask).InferShape(InferShape4NMSWithMask);
// ---------------- NMSWithMask Op END---------------------
} //  namespace ops
