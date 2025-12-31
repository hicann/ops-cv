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

static Status ParseParamsGridSample(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    bool align_corners = false;
    std::string mode_value;
    std::string padding_mode_value;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "align_corners" && attr.i() != 0) {
            if (attr.i() > 1 || attr.i() < -1) {
                OP_LOGE(GetOpName(op_dest).c_str(), "the value of align_corners out of range");
            }
            align_corners = true;
        }
        if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            mode_value = attr.s();
        }
        if (attr.name() == "padding_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            padding_mode_value = attr.s();
        }
    }

    auto orgTensorX = op_dest.GetInputDescByName("x");
    orgTensorX.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorX.SetFormat(ge::FORMAT_NCHW);
    op_dest.UpdateInputDesc("x", orgTensorX);

    auto orgTensorY = op_dest.GetInputDescByName("y");
    orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorY.SetFormat(ge::FORMAT_NCHW);
    op_dest.UpdateOutputDesc("y", orgTensorY);

    op_dest.SetAttr("interpolation_mode", mode_value);
    op_dest.SetAttr("align_corners", align_corners);
    op_dest.SetAttr("padding_mode", padding_mode_value);

    return SUCCESS;
}

static Status ParseParamsGridSampleV2(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    bool align_corners = false;
    std::string mode_value;
    std::string padding_mode_value;
    bool channel_last = false;
    int scheduler_mode = 0;
    for (const auto& attrv2 : node->attribute()) {
        if (attrv2.name() == "align_corners" && attrv2.i() != 0) {
            if (attrv2.i() > 1 || attrv2.i() < -1) {
                OP_LOGE(GetOpName(op_dest).c_str(), "the value of align_corners out of range");
            }
            align_corners = true;
        }
        if (attrv2.name() == "mode" && attrv2.type() == ge::onnx::AttributeProto::STRING) {
            mode_value = attrv2.s();
        }
        if (attrv2.name() == "padding_mode" && attrv2.type() == ge::onnx::AttributeProto::STRING) {
            padding_mode_value = attrv2.s();
        }
        if (attrv2.name() == "channel_last" && attrv2.i() != 0) {
            if (attrv2.i() > 1 || attrv2.i() < 0) {
                OP_LOGE(GetOpName(op_dest).c_str(), "the value of channel_last out of range");
            }
            channel_last = true;
        }
        if (attrv2.name() == "scheduler_mode" && attrv2.type() == ge::onnx::AttributeProto::INT) {
            scheduler_mode = attrv2.i();
        }
    }

    op_dest.SetAttr("interpolation_mode", mode_value);
    op_dest.SetAttr("align_corners", align_corners);
    op_dest.SetAttr("padding_mode", padding_mode_value);
    op_dest.SetAttr("channel_last", channel_last);
    op_dest.SetAttr("scheduler_mode", scheduler_mode);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("GridSample")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::11::GridSampleV2"),
                 ge::AscendString("ai.onnx::16::GridSampleV2"),
                 ge::AscendString("ai.onnx::17::GridSampleV2"),
                 ge::AscendString("ai.onnx::18::GridSampleV2")})
  .ParseParamsFn(ParseParamsGridSampleV2)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("GridSampler2D")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::11::GridSample"),
                 ge::AscendString("ai.onnx::16::GridSample"),
                 ge::AscendString("ai.onnx::17::GridSample"),
                 ge::AscendString("ai.onnx::18::GridSample")})
  .ParseParamsFn(ParseParamsGridSample)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("GridSampler3D")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::11::GridSample3D"),
		             ge::AscendString("ai.onnx::16::GridSample3D"),
		             ge::AscendString("ai.onnx::17::GridSample3D"),
		             ge::AscendString("ai.onnx::18::GridSample3D")})
  .ParseParamsFn(ParseParamsGridSample)
  .ImplyType(ImplyType::TVM);
} // namespace domi