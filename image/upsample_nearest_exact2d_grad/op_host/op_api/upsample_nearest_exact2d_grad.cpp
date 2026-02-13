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
 * \file upsample_nearest_exact2d_grad.cpp
 * \brief
 */
#include "upsample_nearest_exact2d_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearestExact2dGrad);
OP_TYPE_REGISTER(UpsampleNearest2dGrad);

static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;

const aclTensor* UpsampleNearestExact2dGrad(
    const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize, aclTensor* output,
    float scales_h, float scales_w, bool isExact, aclOpExecutor* executor)
{
    L0_DFX(UpsampleNearestExact2dGrad, gradOutput, outputSize, inputSize, output, scales_h, scales_w);

    Shape gradOutputStorageShape = gradOutput->GetStorageShape();
    Shape gradOutputOriginalShape = gradOutput->GetOriginalShape();
    Format gradOutputStorageFormat = gradOutput->GetStorageFormat();
    Format gradOutputOriginalFormat = gradOutput->GetOriginalFormat();

    gradOutputStorageShape.SetDim(DIM_H, (*inputSize)[DIM_H]);
    gradOutputStorageShape.SetDim(DIM_W, (*inputSize)[DIM_W]);
    gradOutputOriginalShape.SetDim(DIM_H, (*inputSize)[DIM_H]);
    gradOutputOriginalShape.SetDim(DIM_W, (*inputSize)[DIM_W]);

    const aclTensor* out = executor->AllocTensor(
        gradOutputStorageShape,
        gradOutputOriginalShape,
        gradOutput->GetDataType(),
        gradOutputStorageFormat,
        gradOutputOriginalFormat);
    CHECK_RET(out != nullptr, nullptr);

    if (isExact) {
        ADD_TO_LAUNCHER_LIST_AICORE(
            UpsampleNearestExact2dGrad, OP_INPUT(gradOutput), OP_OUTPUT(out),
            OP_ATTR(outputSize, inputSize, scales_h, scales_w));
    } else {
        ADD_TO_LAUNCHER_LIST_AICORE(
            UpsampleNearest2dGrad, OP_INPUT(gradOutput), OP_OUTPUT(out),
            OP_ATTR(outputSize, inputSize, scales_h, scales_w));
    }
    return out;
}
} // namespace l0op
