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
 * \file resize_bilinear_v2_grad.cpp
 * \brief resize_bilinear_v2_grad
 */
#include "resize_bilinear_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeBilinearV2Grad);

static aclTensor *ResizeBilinearV2Grad5HdAICORE(const aclTensor *gradOut, const aclTensor *image, bool alignCorners,
    bool halfPixelCenters, aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2Grad5HdAICORE, gradOut, image, alignCorners, halfPixelCenters, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeBilinearV2Grad, OP_INPUT(gradOut, image), OP_OUTPUT(out), OP_ATTR(alignCorners, halfPixelCenters));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeBilinearV2Grad5HdAICORE ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
const aclTensor *ResizeBilinearV2Grad5Hd(
    const aclTensor *gradOut, const aclTensor *image, bool alignCorners, bool halfPixelCenters, aclOpExecutor *executor)
{
    auto out = executor->AllocTensor(image->GetStorageShape(),
        image->GetOriginalShape(),
        image->GetDataType(),
        image->GetStorageFormat(),
        image->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);

    return ResizeBilinearV2Grad5HdAICORE(gradOut, image, alignCorners, halfPixelCenters, out, executor);
}

static aclTensor *ResizeBilinearV2GradAiCore(const aclTensor *grads, const aclTensor *originalImage,
    const bool alignCorners, const bool halfPixelCenters, const aclFloatArray *scales, aclTensor *y,
    aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2GradAiCore, grads, originalImage, alignCorners, halfPixelCenters, scales, y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeBilinearV2Grad,
        OP_INPUT(grads, originalImage),
        OP_OUTPUT(y),
        OP_ATTR(alignCorners, halfPixelCenters, scales));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeBilinearV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}

const aclTensor *ResizeBilinearV2Grad(const aclTensor *grads, const aclTensor *originalImage, const bool alignCorners,
    const bool halfPixelCenters, const aclFloatArray *scales, aclOpExecutor *executor)
{
    auto y = executor->AllocTensor(
        originalImage->GetViewShape(), originalImage->GetDataType(), originalImage->GetViewFormat());
    CHECK_RET(y != nullptr, nullptr);

    return ResizeBilinearV2GradAiCore(grads, originalImage, alignCorners, halfPixelCenters, scales, y, executor);
}
}  // namespace l0op
