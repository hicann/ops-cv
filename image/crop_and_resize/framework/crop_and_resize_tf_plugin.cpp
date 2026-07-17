/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"

#include "log/log.h"

namespace domi {
static Status ParseCropAndResize(const ge::Operator& op_src, ge::Operator& op)
{
    (void)AutoMappingByOpFn(op_src, op);

    ge::TensorDesc input_tensor = op.GetInputDesc("x");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc("x", input_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "update input format failed.");
        return FAILED;
    }
    ge::TensorDesc output_tensor = op.GetOutputDesc("y");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op.UpdateOutputDesc("y", output_tensor);
    if (ret_output != ge::GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
}
// register CropAndResize op to GE
REGISTER_CUSTOM_OP("CropAndResize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CropAndResize")
    .ParseParamsByOperatorFn(ParseCropAndResize)
    .ImplyType(ImplyType::TVM);
} // namespace domi
