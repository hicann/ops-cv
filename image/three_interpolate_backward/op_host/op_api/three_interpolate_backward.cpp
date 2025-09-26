/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "three_interpolate_backward.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

namespace {
constexpr uint8_t DIM_INDEX_B = 0;
constexpr uint8_t DIM_INDEX_C1 = 1;
constexpr uint8_t DIM_INDEX_C = 1;
constexpr uint8_t CONST_1 = 1;
constexpr uint8_t C0 = 16;
} // namespace

namespace l0op {
OP_TYPE_REGISTER(ThreeInterpolateBackward);

const aclTensor* ThreeInterpolateBackwardAicore(
    const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, int m, aclTensor* grad_y,
    aclOpExecutor* executor)
{
    L0_DFX(ThreeInterpolateBackwardAicore, grad_x, idx, weight, m, grad_y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ThreeInterpolateBackward, OP_INPUT(grad_x, idx, weight), OP_OUTPUT(grad_y), OP_ATTR(m));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ThreeInterpolateBackwardAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return grad_y;
}

void ThreeInterpolateBackwardInferShapeStorage(const Shape& grad_x_shape, int m, Shape& grad_y_shape)
{
    grad_y_shape.AppendDim(grad_x_shape.GetDim(DIM_INDEX_B));  // N
    grad_y_shape.AppendDim(grad_x_shape.GetDim(DIM_INDEX_C1)); // C1
    grad_y_shape.AppendDim(m);                                 // H
    grad_y_shape.AppendDim(CONST_1);                           // W = 1
    grad_y_shape.AppendDim(C0);                                // C0 = 16
}

void ThreeInterpolateBackwardInferShapeView(const Shape& grad_x_shape, int m, Shape& grad_y_shape)
{
    grad_y_shape.AppendDim(grad_x_shape.GetDim(DIM_INDEX_B)); // B
    grad_y_shape.AppendDim(grad_x_shape.GetDim(DIM_INDEX_C)); // C
    grad_y_shape.AppendDim(m);                                // M
    grad_y_shape.AppendDim(CONST_1);                          // 1
}

const aclTensor* ThreeInterpolateBackward(
    const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, int m, aclOpExecutor* executor)
{
    Shape grad_y_storage_shape;
    auto dtype = grad_x->GetDataType();
    auto grad_y_storage_format = grad_x->GetStorageFormat();
    ThreeInterpolateBackwardInferShapeStorage(grad_x->GetStorageShape(), m, grad_y_storage_shape);

    Shape grad_y_view_shape;
    auto grad_y_view_format = op::Format::FORMAT_ND;
    ThreeInterpolateBackwardInferShapeView(grad_x->GetViewShape(), m, grad_y_view_shape);

    auto grad_y = executor->AllocTensor(
        grad_y_storage_shape, grad_y_view_shape, dtype, grad_y_storage_format, grad_y_view_format);
    CHECK_RET(grad_y != nullptr, nullptr);

    return ThreeInterpolateBackwardAicore(grad_x, idx, weight, m, grad_y, executor);
}
} // namespace l0op