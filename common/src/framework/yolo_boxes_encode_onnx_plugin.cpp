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
 * \file yolo_boxes_encode_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsYoloBoxesEncode(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::string performance_mode = "high_precision";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "performance_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            performance_mode = attr.s();
        }
    }

    op_dest.SetAttr("performance_mode", performance_mode);
    return SUCCESS;
}

// register YoloBoxesEncode op info to GE
REGISTER_CUSTOM_OP("YoloBoxesEncode")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::NPUYoloBoxesEncode"), ge::AscendString("ai.onnx::12::NPUYoloBoxesEncode"),
         ge::AscendString("ai.onnx::13::NPUYoloBoxesEncode"), ge::AscendString("ai.onnx::14::NPUYoloBoxesEncode"),
         ge::AscendString("ai.onnx::15::NPUYoloBoxesEncode"), ge::AscendString("ai.onnx::16::NPUYoloBoxesEncode"),
         ge::AscendString("ai.onnx::17::NPUYoloBoxesEncode"), ge::AscendString("ai.onnx::18::NPUYoloBoxesEncode"),
         ge::AscendString("npu::1::NPUYoloBoxesEncode")})
    .ParseParamsFn(ParseParamsYoloBoxesEncode)
    .ImplyType(ImplyType::TVM);
} // namespace domi
