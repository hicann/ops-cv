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
 * \file npu_ps_roi_pooling_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuPsRoiPooling(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float spatial_scale = 0.0;
    int output_dim = 1;
    int group_size = 0;
    int attr_num = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            spatial_scale = attr.f();
            op_dest.SetAttr("spatial_scale", spatial_scale);
            ++attr_num;
        } else if (attr.name() == "output_dim" && attr.type() == ge::onnx::AttributeProto::INT) {
            output_dim = attr.i();
            op_dest.SetAttr("output_dim", output_dim);
            ++attr_num;
        } else if (attr.name() == "group_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            group_size = attr.i();
            op_dest.SetAttr("group_size", group_size);
            ++attr_num;
        }
    }
    // Node must have 3 required attr.
    if (attr_num != 3) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr spatial_scale, output_dim, group_size");
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PSROIPoolingV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::11::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::12::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::13::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::14::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::15::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::16::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::17::NPUPsRoiPooling"),
                 ge::AscendString("ai.onnx::18::NPUPsRoiPooling")})
  .ParseParamsFn(ParseParamsNpuPsRoiPooling)
  .ImplyType(ImplyType::TVM);
} // namespace domi