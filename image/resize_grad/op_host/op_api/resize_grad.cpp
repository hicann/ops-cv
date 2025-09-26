/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_grad.cpp
 * \brief
 */
#include "resize_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeGrad);
static const float CUBIC_COEFF_A = -0.75f;
static constexpr size_t DIM_W = 3;
static const std::string HALF_PIXEL = "half_pixel";
static const std::string MODE = "nearest";
static const std::string NEAREST_MODE = "floor";

const aclTensor *ResizeGrad(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclTensor *scales,
    const aclTensor *sizes, aclOpExecutor *executor)
{
    L0_DFX(ResizeGrad, gradOutput, scales, sizes);
    const int64_t inputSizeW = (*inputSize)[DIM_W];
    op::Shape gradsStorageShape = gradOutput->GetStorageShape();
    op::Shape gradsOriginalShape = gradOutput->GetOriginalShape();
    gradsStorageShape.SetDim(DIM_W, inputSizeW);
    gradsOriginalShape.SetDim(DIM_W, inputSizeW);

    auto out = executor->AllocTensor(gradsStorageShape,
        gradsOriginalShape,
        gradOutput->GetDataType(),
        gradOutput->GetStorageFormat(),
        gradOutput->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);
    static internal::AicpuTaskSpace space("ResizeGrad");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ResizeGrad,
        OP_ATTR_NAMES({"coordinate_transformation_mode", "cubic_coeff_a", "mode", "nearest_mode"}),
        OP_INPUT(gradOutput, scales, scales, sizes),
        OP_OUTPUT(out),
        OP_ATTR(HALF_PIXEL, CUBIC_COEFF_A, MODE, NEAREST_MODE));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}
}  // namespace l0op
