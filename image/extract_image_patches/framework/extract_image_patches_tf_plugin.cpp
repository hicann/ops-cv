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
static Status ExtractImagePatchesMappingFn(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "AutoMappingByOpFn failed.");
        return FAILED;
    }

    ge::TensorDesc tensor_desc_w = op.GetInputDesc(0);
    ge::TensorDesc tensor_desc_w1 = op.GetOutputDesc(0);
    tensor_desc_w.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_desc_w1.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_desc_w.SetFormat(ge::FORMAT_NHWC);
    tensor_desc_w1.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc(static_cast<uint32_t>(0), tensor_desc_w);
    auto ret1 = op.UpdateOutputDesc(static_cast<uint32_t>(0), tensor_desc_w1);
    if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "UpdateInputDesc or UpdateOutputDesc failed.");
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ExtractImagePatches")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ExtractImagePatches")
    .ParseParamsByOperatorFn(ExtractImagePatchesMappingFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
