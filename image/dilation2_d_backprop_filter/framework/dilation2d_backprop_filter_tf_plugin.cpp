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
#include "cv_plugin_util.h"

namespace domi {
static Status ParseDilation2DBackpropFilter(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "AutoMappingByOpFn failed.");
        return FAILED;
    }

    ge::TensorDesc input_tensor = op.GetInputDesc("x");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc("x", input_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update input format failed.");
        return FAILED;
    }

    ge::TensorDesc filter_tensor = op.GetInputDesc("filter");
    filter_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    filter_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_filter = op.UpdateInputDesc("filter", filter_tensor);
    if (ret_filter != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update filter format failed.");
        return FAILED;
    }

    ge::TensorDesc out_backprop_tensor = op.GetInputDesc("out_backprop");
    out_backprop_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    out_backprop_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_out_backprop = op.UpdateInputDesc("out_backprop", out_backprop_tensor);
    if (ret_out_backprop != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update out_backprop format failed.");
        return FAILED;
    }

    ge::TensorDesc output_tensor = op.GetOutputDesc("y");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op.UpdateOutputDesc("y", output_tensor);
    if (ret_output != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update output format failed.");
        return FAILED;
    }

    std::string padding;
    if (op.GetAttr("padding", padding) == ge::GRAPH_SUCCESS) {
        (void)op.SetAttr("padding_mode", padding);
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Dilation2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Dilation2DBackpropFilter")
    .ParseParamsByOperatorFn(ParseDilation2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
} // namespace domi
