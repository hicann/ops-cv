/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_linear_infershape.cpp
 * \brief resize_linear_infershape
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t IN_SIZE = 1;
static constexpr size_t OUT_Y = 0;
static constexpr size_t IN_X_DIMS = 3;
static constexpr int64_t IN_SIZE_NUM = 1;
static constexpr size_t IDX_L = 2;

ge::graphStatus ResizeLinearInferShape(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeLinearInferShape rt2.0");

    auto nodeName = context->GetNodeName();

    auto xShape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto sizeTensor = context->GetInputTensor(IN_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);
    auto sizeDtype = sizeTensor->GetDataType();
    OP_CHECK_IF(sizeDtype != ge::DT_INT32, OP_LOGE(nodeName, "size dtype only support int32"), return GRAPH_FAILED);
    int32_t outL = ge::UNKNOWN_DIM;
    if (Ops::Cv::IsConstTensor(sizeTensor)) {
        const int32_t* sizeValue = sizeTensor->GetData<int32_t>();
        auto sizeNum = sizeTensor->GetShapeSize();
        OP_CHECK_IF(sizeNum != IN_SIZE_NUM,
            OP_LOGE(nodeName, "the element number of size should be 1, but is %ld", sizeNum), return GRAPH_FAILED);
        outL = sizeValue[0];
        OP_CHECK_IF(outL <= 0, OP_LOGE(nodeName, "output size should be greater than 0"), return GRAPH_FAILED);
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownShape(IN_X_DIMS, *yShape);
    } else {
        OP_CHECK_IF(
            xShape->GetDimNum() != IN_X_DIMS, OP_LOGE(nodeName, "x shape only support 3D"), return GRAPH_FAILED);
        OP_CHECK_IF(
            (xShape->GetDim(IDX_L) == 0) || (xShape->GetDim(IDX_L - 1) == 0),
            OP_LOGE(nodeName, "input size should be greater than 0"), return GRAPH_FAILED);
        *yShape = *xShape;
    }

    yShape->SetDim(IDX_L, outL);

    OP_LOGI(context->GetNodeName(), "End to do ResizeLinearInferShape rt2.0");
    return ge::GRAPH_SUCCESS;
}

graphStatus ResizeLinearInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeLinearInferDtype");

    auto xDtype = context->GetInputDataType(IN_X);
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT) && (xDtype != ge::DT_FLOAT16) && (xDtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "x dtype only support float, float16 and bfloat16"), return GRAPH_FAILED);

    context->SetOutputDataType(OUT_Y, xDtype);

    OP_LOGI(context->GetNodeName(), "End to do ResizeLinearInferDtype");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeLinear)
    .InferShape(ResizeLinearInferShape)
    .InputsDataDependency({IN_SIZE})
    .InferDataType(ResizeLinearInferDtype);
} // namespace ops
