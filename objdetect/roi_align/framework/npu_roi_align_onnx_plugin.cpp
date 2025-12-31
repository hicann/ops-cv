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
// required attr judge
static const int req_attr_num = 3;
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsNpuRoiAlign(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    float spatial_scale = 0.0;
    int pooled_height = 0;
    int pooled_width = 0;
    int sample_num = 2;
    int roi_end_mode = 1;
    // The initialization of required attributes count
    int req_attr_count = 0;
    for (auto attr : node->attribute()) {
        if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            spatial_scale = attr.f();
            op_dest.SetAttr("spatial_scale", spatial_scale);
            ++req_attr_count;
        } else if (attr.name() == "pooled_height" && attr.type() == ge::onnx::AttributeProto::INT) {
            pooled_height = attr.i();
            op_dest.SetAttr("pooled_height", pooled_height);
            ++req_attr_count;
        } else if (attr.name() == "pooled_width" && attr.type() == ge::onnx::AttributeProto::INT) {
            pooled_width = attr.i();
            op_dest.SetAttr("pooled_width", pooled_width);
            ++req_attr_count;
        } else if (attr.name() == "sample_num" && attr.type() == ge::onnx::AttributeProto::INT) {
            sample_num = attr.i();
        } else if (attr.name() == "roi_end_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
            roi_end_mode = attr.i();
        }
    }
    // Node must have 3 required attributes.
    if (req_attr_count != req_attr_num) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr spatial_scale, pooled_height, pooled_width");
        return FAILED;
    }
    op_dest.SetAttr("sample_num", sample_num);
    op_dest.SetAttr("roi_end_mode", roi_end_mode);

    return SUCCESS;
}

// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("ROIAlign")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPURoiAlign"),
                   ge::AscendString("ai.onnx::11::NPURoiAlign"),
                   ge::AscendString("ai.onnx::12::NPURoiAlign"),
                   ge::AscendString("ai.onnx::13::NPURoiAlign"),
                   ge::AscendString("ai.onnx::14::NPURoiAlign"),
                   ge::AscendString("ai.onnx::15::NPURoiAlign"),
                   ge::AscendString("ai.onnx::16::NPURoiAlign"),
                   ge::AscendString("ai.onnx::17::NPURoiAlign"),
                   ge::AscendString("ai.onnx::18::NPURoiAlign")})
    .ParseParamsFn(ParseParamsNpuRoiAlign)
    .ImplyType(ImplyType::TVM);
}  // namespace domi