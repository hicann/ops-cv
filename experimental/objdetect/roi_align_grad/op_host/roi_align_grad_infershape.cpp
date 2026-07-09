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
 * \file roi_align_grad_infershape.cpp
 * \brief RoiAlignGrad InferShape implementation.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t kAttrXdiffShape = 0U;
constexpr size_t kYDiffIndex = 0U;
constexpr size_t kXDiffIndex = 0U;
constexpr int64_t kC0Size = 16;

static int64_t CeilDiv(int64_t value, int64_t divisor) { return divisor == 0 ? 0 : (value + divisor - 1) / divisor; }
} // namespace

namespace ops {

static ge::graphStatus InferShapeRoiAlignGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRoiAlignGrad");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* xdiffShape = attrs->GetListInt(kAttrXdiffShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xdiffShape);

    gert::Shape* yShape = context->GetOutputShape(kXDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const size_t dimNum = xdiffShape->GetSize();
    const int64_t* dims = xdiffShape->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, dims);

    OP_CHECK_IF(dimNum != 4U, OP_LOGE(context->GetNodeName(), "xdiff_shape must be 4D, but got %zu.", dimNum),
                return GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; ++i) {
        OP_CHECK_IF(dims[i] <= 0,
                    OP_LOGE(context->GetNodeName(), "xdiff_shape[%zu] must be positive, but got %lld.", i,
                            static_cast<long long>(dims[i])),
                    return GRAPH_FAILED);
    }

    const gert::Shape* yDiffShape = context->GetInputShape(kYDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDiffShape);

    if (yDiffShape->GetDimNum() == 4U) {
        yShape->SetDimNum(4U);
        yShape->SetDim(0U, dims[0]);
        yShape->SetDim(1U, dims[1]);
        yShape->SetDim(2U, dims[2]);
        yShape->SetDim(3U, dims[3]);
    } else {
        yShape->SetDimNum(5U);
        yShape->SetDim(0U, dims[0]);
        yShape->SetDim(1U, CeilDiv(dims[1], kC0Size));
        yShape->SetDim(2U, dims[2]);
        yShape->SetDim(3U, dims[3]);
        yShape->SetDim(4U, kC0Size);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeRoiAlignGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiAlignGrad).InferShape(InferShapeRoiAlignGrad);
} // namespace ops
