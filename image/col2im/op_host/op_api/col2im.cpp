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
 * \file col2im.cpp
 * \brief
 */

#include "col2im.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Col2im);

const aclTensor *Col2im(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclIntArray *kernelSize,
    const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclOpExecutor *executor)
{
    L0_DFX(Col2im, gradOutput, inputSize, kernelSize, dilation, padding, stride);
    auto dims = executor->ConvertToTensor(inputSize, op::DataType::DT_INT32);
    auto out = executor->AllocTensor(
        gradOutput->GetDataType(), gradOutput->GetStorageFormat(), gradOutput->GetOriginalFormat());
    INFER_SHAPE(Col2im, OP_INPUT(gradOutput, dims), OP_OUTPUT(out), OP_ATTR(kernelSize, dilation, padding, stride));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Col2im, OP_INPUT(gradOutput, dims), OP_OUTPUT(out), OP_ATTR(kernelSize, dilation, padding, stride));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Col2imAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
}  // namespace l0op
