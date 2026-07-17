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
#include "graph/utils/op_desc_utils.h"

namespace domi {
static Status ResizeBilinearV2MappingFn(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        return FAILED;
    }
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc tensor_desc_w = op_dsc->GetInputDesc(0);
    ge::GeTensorDesc tensor_desc_w1 = op_dsc->GetOutputDesc(0);
    tensor_desc_w.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_desc_w1.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_desc_w.SetFormat(ge::FORMAT_NHWC);
    tensor_desc_w1.SetFormat(ge::FORMAT_NHWC);
    auto ret = op_dsc->UpdateInputDesc(0, tensor_desc_w);
    auto ret1 = op_dsc->UpdateOutputDesc(0, tensor_desc_w1);
    if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ResizeBilinearV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeBilinear")
    .ParseParamsByOperatorFn(ResizeBilinearV2MappingFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
