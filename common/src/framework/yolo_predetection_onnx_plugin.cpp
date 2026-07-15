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
 * \file yolo_predetection_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsYoloPreDetection(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int boxes = 3;
    int coords = 4;
    int classes = 80;
    std::string yolo_version = "V5";
    bool softmax = false;
    bool background = false;
    bool softmaxtree = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "boxes") {
            boxes = attr.i();
        } else if (attr.name() == "coords") {
            coords = attr.i();
        } else if (attr.name() == "classes") {
            classes = attr.i();
        } else if (attr.name() == "yolo_version") {
            yolo_version = attr.s();
        } else if (attr.name() == "softmax") {
            softmax = attr.i();
        } else if (attr.name() == "background") {
            background = attr.i();
        } else if (attr.name() == "softmaxtree") {
            softmaxtree = attr.i();
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

// register YoloPreDetection op info to GE
REGISTER_CUSTOM_OP("YoloPreDetection")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::YoloPreDetection"), ge::AscendString("ai.onnx::9::YoloPreDetection"),
                   ge::AscendString("ai.onnx::10::YoloPreDetection"), ge::AscendString("ai.onnx::11::YoloPreDetection"),
                   ge::AscendString("ai.onnx::12::YoloPreDetection"), ge::AscendString("ai.onnx::13::YoloPreDetection"),
                   ge::AscendString("ai.onnx::14::YoloPreDetection"), ge::AscendString("ai.onnx::15::YoloPreDetection"),
                   ge::AscendString("ai.onnx::16::YoloPreDetection")})
    .ParseParamsFn(ParseParamsYoloPreDetection)
    .ImplyType(ImplyType::TVM);
} // namespace domi
