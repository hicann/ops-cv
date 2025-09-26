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
 * \file upsample_bilinear2d_grad.cpp
 * \brief
 */
#include "upsample_bilinear2d_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBilinear2dGrad);

const aclTensor *UpsampleBilinear2dGrad(const aclTensor *gradOutput, const aclIntArray *outputSize,
    const aclIntArray *inputSize, const aclTensor *output, const bool alignCorners, const float scales_h,
    const float scales_w, aclOpExecutor *executor)
{
    L0_DFX(UpsampleBilinear2dGrad, gradOutput, outputSize, inputSize, output, alignCorners, scales_h, scales_w);

    const aclTensor *out =
        executor->AllocTensor(output->GetViewShape(), gradOutput->GetDataType(), output->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBilinear2dGrad,
        OP_INPUT(gradOutput),
        OP_OUTPUT(out),
        OP_ATTR(outputSize, inputSize, alignCorners, scales_h, scales_w));

    return out;
}
}  // namespace l0op
