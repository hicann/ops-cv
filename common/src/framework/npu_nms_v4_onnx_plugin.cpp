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
 * \file npu_nms_v4_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "op_cv_proto_extend.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNmsV4(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    op_dest.SetAttr("original_type", "npu::1::NPUNmsV4");

    bool pad_to_max_output_size = false;
    int max_output_size = 0;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "pad_to_max_output_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() == 1) {
                pad_to_max_output_size = true;
            }
        } else if (attr.name() == "max_output_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            max_output_size = attr.i();
        }
    }

    ge::Tensor scalar_const_value = CreateScalar(max_output_size, ge::DT_INT32);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("pad_to_max_output_size", pad_to_max_output_size);
    op_dest.SetAttr("max_output_size", scalar_const_value);

    return SUCCESS;
}

static Status ParseOpToGraphNmsV4(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    bool pad_to_max_output_size = false;
    op.GetAttr("pad_to_max_output_size", pad_to_max_output_size);

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);

    ge::Tensor const_value;
    if (op.GetAttr("max_output_size", const_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get max_output_size from op failed");
        return FAILED;
    }

    auto const_op = ge::op::Const((ori_name + "_const_data").c_str()).set_attr_value(const_value);
    auto NonMaxSuppressionV4 = ge::op::NonMaxSuppressionV4((ori_name + "_NonMaxSuppressionV4").c_str())
                                   .set_input_boxes(data0)
                                   .set_input_scores(data1)
                                   .set_input_max_output_size(const_op)
                                   .set_input_iou_threshold(data2)
                                   .set_input_score_threshold(data3)
                                   .set_attr_pad_to_max_output_size(pad_to_max_output_size);

    std::vector<ge::Operator> inputs{data0, data1, data2, data3};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(NonMaxSuppressionV4, std::vector<std::size_t>{0});
    outputs.emplace_back(NonMaxSuppressionV4, std::vector<std::size_t>{1});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register NonMaxSuppressionV4 op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUNmsV4"),
                 ge::AscendString("ai.onnx::11::NPUNmsV4"),
                 ge::AscendString("ai.onnx::12::NPUNmsV4"),
                 ge::AscendString("ai.onnx::13::NPUNmsV4"),
                 ge::AscendString("ai.onnx::14::NPUNmsV4"),
                 ge::AscendString("ai.onnx::15::NPUNmsV4"),
                 ge::AscendString("ai.onnx::16::NPUNmsV4"),
                 ge::AscendString("ai.onnx::17::NPUNmsV4"),
                 ge::AscendString("ai.onnx::18::NPUNmsV4")})
  .ParseParamsFn(ParseParamsNmsV4)
  .ParseOpToGraphFn(ParseOpToGraphNmsV4)
  .ImplyType(ImplyType::TVM);
} // namespace domi