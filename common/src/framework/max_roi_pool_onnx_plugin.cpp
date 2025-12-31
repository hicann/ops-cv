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
 * \file max_roi_pool_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
static const size_t POOLED_SHAPE_SIZE = 2;

static Status ParseParamsMaxRoiPool(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    float spatial_scale = 1.0;
    std::vector<int> pooled_shape;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "pooled_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                pooled_shape.push_back(attr.ints(i));
            }
        } else if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            spatial_scale = static_cast<float>(attr.f());
        }
    }
    if (pooled_shape.size() != POOLED_SHAPE_SIZE) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Obtain attr pooled_shape failed.");
        return FAILED;
    }

    op_dest.SetAttr("pooled_h", pooled_shape[0]);
    op_dest.SetAttr("pooled_w", pooled_shape[1]);
    op_dest.SetAttr("spatial_scale_h", spatial_scale);
    op_dest.SetAttr("spatial_scale_w", spatial_scale);
    return SUCCESS;
}

//register MaxRoiPool op info to GE
REGISTER_CUSTOM_OP("ROIPooling")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::1::MaxRoiPool"),
                   ge::AscendString("ai.onnx::8::MaxRoiPool"),
                   ge::AscendString("ai.onnx::9::MaxRoiPool"),
                   ge::AscendString("ai.onnx::10::MaxRoiPool"),
                   ge::AscendString("ai.onnx::11::MaxRoiPool"),
                   ge::AscendString("ai.onnx::12::MaxRoiPool"),
                   ge::AscendString("ai.onnx::13::MaxRoiPool"),
                   ge::AscendString("ai.onnx::14::MaxRoiPool"),
                   ge::AscendString("ai.onnx::15::MaxRoiPool"),
                   ge::AscendString("ai.onnx::16::MaxRoiPool"),
                   ge::AscendString("ai.onnx::17::MaxRoiPool"),
                   ge::AscendString("ai.onnx::18::MaxRoiPool")})
    .ParseParamsFn(ParseParamsMaxRoiPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
