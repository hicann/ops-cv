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
 * \file npu_batch_nms_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsBatchNMS(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float iou_threshold = 0.0;
    float score_threshold = 0.0;
    int max_size_per_class = 0;
    int max_total_size = 0;
    bool change_coordinate_frame = false;
    bool transpose_box = false;

    for (auto attr : node->attribute()) {
        if (attr.name() == "iou_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            iou_threshold = attr.f();
            op_dest.SetAttr("iou_threshold", iou_threshold);
        } else if (attr.name() == "score_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            score_threshold = attr.f();
            op_dest.SetAttr("score_threshold", score_threshold);
        } else if (attr.name() == "max_size_per_class" && attr.type() == ge::onnx::AttributeProto::INT) {
            max_size_per_class = attr.i();
            op_dest.SetAttr("max_size_per_class", max_size_per_class);
        } else if (attr.name() == "max_total_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            max_total_size = attr.i();
            op_dest.SetAttr("max_total_size", max_total_size);
        } else if (attr.name() == "change_coordinate_frame" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() == 1) {
                change_coordinate_frame = true;
            }
            op_dest.SetAttr("change_coordinate_frame", change_coordinate_frame);
        } else if (attr.name() == "transpose_box" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() == 1) {
                transpose_box = true;
            }
            op_dest.SetAttr("transpose_box", transpose_box);
        }
    }

    return SUCCESS;
}

// register StrideAdd op info to GE
REGISTER_CUSTOM_OP("BatchMultiClassNonMaxSuppression")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUBatchNms"),
                 ge::AscendString("ai.onnx::11::NPUBatchNms"),
                 ge::AscendString("ai.onnx::12::NPUBatchNms"),
                 ge::AscendString("ai.onnx::13::NPUBatchNms"),
                 ge::AscendString("ai.onnx::14::NPUBatchNms"),
                 ge::AscendString("ai.onnx::15::NPUBatchNms"),
                 ge::AscendString("ai.onnx::16::NPUBatchNms"),
                 ge::AscendString("ai.onnx::17::NPUBatchNms"),
                 ge::AscendString("ai.onnx::18::NPUBatchNms")})
  .ParseParamsFn(ParseParamsBatchNMS)
  .ImplyType(ImplyType::TVM);
} // namespace domi