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
 * \file resize_bicubic_v2_grad.cpp
 * \brief resize_bicubic_v2_grad
 */

#include "resize_bicubic_v2_grad.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeBicubicV2Grad);

static const aclTensor *ResizeBicubicV2GradAICORE(const aclTensor *gradOut, const aclTensor *image, bool alignCorners,
    const aclFloatArray *scales, aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(ResizeBicubicV2GradAICORE, gradOut, image, alignCorners, scales, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeBicubicV2Grad, OP_INPUT(gradOut, image), OP_OUTPUT(y), OP_ATTR(alignCorners, scales));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeBicubicV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);

    return y;
}

const aclTensor *ResizeBicubicV2Grad(const aclTensor *grads, const aclTensor *originalImage, const bool alignCorners,
    const aclFloatArray *scales, const aclTensor *out, aclOpExecutor *executor)
{
    auto y = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), out->GetViewFormat());
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc resize_bicubic_v2_grad out tensor failed");
        return nullptr;
    }

    return ResizeBicubicV2GradAICORE(grads, originalImage, alignCorners, scales, y, executor);
}

}  // namespace l0op
