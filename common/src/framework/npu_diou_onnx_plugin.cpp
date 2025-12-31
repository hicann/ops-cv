/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuDiou(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool trans = false;
    bool is_cross = false;
    std::string mode = "iou";

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "trans" && attr.type() == ge::onnx::AttributeProto::INT) {
            trans = (attr.i() == 1);
        } else if (attr.name() == "is_cross" && attr.type() == ge::onnx::AttributeProto::INT) {
            is_cross = (attr.i() == 1);
        } else if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            mode = "iof";
        }
    }

    op_dest.SetAttr("trans", trans);
    op_dest.SetAttr("is_cross", is_cross);
    op_dest.SetAttr("mode", mode);
    return SUCCESS;
}

// register npu_diou op info to GE
REGISTER_CUSTOM_OP("DIoU")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUDiou"), 
                 ge::AscendString("ai.onnx::11::NPUDiou"),
                 ge::AscendString("ai.onnx::12::NPUDiou"),
                 ge::AscendString("ai.onnx::13::NPUDiou"),
                 ge::AscendString("ai.onnx::14::NPUDiou"),
                 ge::AscendString("ai.onnx::15::NPUDiou"),
                 ge::AscendString("ai.onnx::16::NPUDiou"),
                 ge::AscendString("ai.onnx::17::NPUDiou"),
                 ge::AscendString("ai.onnx::18::NPUDiou")})
  .ParseParamsFn(ParseParamsNpuDiou)
  .ImplyType(ImplyType::TVM);
} // namespace domi