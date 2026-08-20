/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "gaussian_blur_utils.h"

namespace ops {

static constexpr size_t KSIZE_ATTR_INDEX = 0;
static constexpr size_t SIGMA_X_ATTR_INDEX = 1;
static constexpr size_t SIGMA_Y_ATTR_INDEX = 2;
static constexpr size_t BORDER_TYPE_ATTR_INDEX = 3;

static bool IsSupportedKernel(int64_t kernel)
{
    return kernel == 1 || kernel == 3 || kernel == 5 || kernel == 7 || kernel == 9 || kernel == 11 || kernel == 15 ||
           kernel == 21 || kernel == 31;
}

static ge::graphStatus InferShapeGaussianBlur(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_IF(inputShape->GetDimNum() < 2U || inputShape->GetDimNum() > 3U,
                OP_LOGE(context, "GaussianBlur only supports rank 2/3 ND tensors."), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < inputShape->GetDimNum(); ++i) {
        OP_CHECK_IF(inputShape->GetDim(i) <= 0, OP_LOGE(context, "GaussianBlur does not support empty tensors."),
                    return ge::GRAPH_FAILED);
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* ksize = attrs->GetListInt(KSIZE_ATTR_INDEX);
    const float* sigmaX = attrs->GetFloat(SIGMA_X_ATTR_INDEX);
    const float* sigmaY = attrs->GetFloat(SIGMA_Y_ATTR_INDEX);
    const int64_t* borderType = attrs->GetInt(BORDER_TYPE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaY);
    OP_CHECK_NULL_WITH_CONTEXT(context, borderType);
    OP_CHECK_IF(ksize->GetSize() != 2U, OP_LOGE(context, "ksize must contain 2 elements."), return ge::GRAPH_FAILED);

    const int64_t* kernelData = ksize->GetData();
    const uint32_t height = static_cast<uint32_t>(inputShape->GetDim(0));
    const uint32_t width = static_cast<uint32_t>(inputShape->GetDim(1));
    gaussian_blur::CanonicalParams params;
    OP_CHECK_IF(!gaussian_blur::CanonicalizeParams(kernelData[0], kernelData[1], static_cast<double>(*sigmaX),
                                                   static_cast<double>(*sigmaY), *borderType, width, height, params),
                OP_LOGE(context, "failed to canonicalize GaussianBlur attributes."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedKernel(params.kernelW) || !IsSupportedKernel(params.kernelH),
                OP_LOGE(context, "GaussianBlur supports K1/K3/K5/K7/K9/K11/K15/K21/K31."), return ge::GRAPH_FAILED);

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GaussianBlur).InferShape(InferShapeGaussianBlur);

} // namespace ops
