/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "resize_linear_grad.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeLinearGrad);

static const aclTensor* ResizeLinearGradAICORE(
    const aclTensor* gradOut, const aclTensor* image, bool alignCorners, const float scale, aclTensor* y,
    aclOpExecutor* executor)
{
    L0_DFX(ResizeLinearGradAICORE, gradOut, image, alignCorners, scale, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeLinearGrad, OP_INPUT(gradOut, image), OP_OUTPUT(y), OP_ATTR(alignCorners, scale));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeLinearGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);

    return y;
}

const aclTensor* ResizeLinearGrad(
    const aclTensor* grads, const aclTensor* originalImage, const bool alignCorners, const float scale,
    const aclTensor* out, aclOpExecutor* executor)
{
    auto y = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), out->GetViewFormat());
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc resize_linear_grad out tensor failed");
        return nullptr;
    }

    return ResizeLinearGradAICORE(grads, originalImage, alignCorners, scale, y, executor);
}

} // namespace l0op
