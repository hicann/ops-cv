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
 * \file yolo_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsYolo(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int boxes = 3;
    int coords = 4;
    int classes = 80;
    std::string yolo_version = "V3";
    bool softmax = false;
    bool background = false;
    bool softmaxtree = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "boxes" && attr.type() == ge::onnx::AttributeProto::INT) {
            boxes = attr.i();
        } else if (attr.name() == "coords" && attr.type() == ge::onnx::AttributeProto::INT) {
            coords = attr.i();
        } else if (attr.name() == "classes" && attr.type() == ge::onnx::AttributeProto::INT) {
            classes = attr.i();
        } else if (attr.name() == "yolo_version" && attr.type() == ge::onnx::AttributeProto::STRING) {
            yolo_version = attr.s();
        }
    }

    op_dest.SetAttr("boxes", boxes);
    op_dest.SetAttr("coords", coords);
    op_dest.SetAttr("classes", classes);
    op_dest.SetAttr("yolo_version", yolo_version);
    op_dest.SetAttr("softmax", softmax);
    op_dest.SetAttr("background", background);
    op_dest.SetAttr("softmaxtree", softmaxtree);

    return SUCCESS;
}

// register Yolo op info to GE
REGISTER_CUSTOM_OP("Yolo")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Yolo"), ge::AscendString("ai.onnx::9::Yolo"),
                   ge::AscendString("ai.onnx::10::Yolo"), ge::AscendString("ai.onnx::11::Yolo"),
                   ge::AscendString("ai.onnx::12::Yolo"), ge::AscendString("ai.onnx::13::Yolo"),
                   ge::AscendString("ai.onnx::14::Yolo"), ge::AscendString("ai.onnx::15::Yolo"),
                   ge::AscendString("ai.onnx::16::Yolo")})
    .ParseParamsFn(ParseParamsYolo)
    .ImplyType(ImplyType::TVM);
} // namespace domi
