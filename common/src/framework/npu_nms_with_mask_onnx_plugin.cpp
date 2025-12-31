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
 * \file npu_nms_with_mask_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNMSWithMask(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float iou_threshold = 0.5;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "iou_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            iou_threshold = attr.f();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("iou_threshold", iou_threshold);

    return SUCCESS;
}

// register StrideAdd op info to GE
REGISTER_CUSTOM_OP("NMSWithMask")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::11::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::12::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::13::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::14::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::15::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::16::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::17::NPUNmsWithMask"),
                 ge::AscendString("ai.onnx::18::NPUNmsWithMask")})
  .ParseParamsFn(ParseParamsNMSWithMask)
  .ImplyType(ImplyType::TVM);
} // namespace domi