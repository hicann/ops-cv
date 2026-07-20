/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_rotated_overlaps_onnx_plugin.cpp
 * \brief onnx plugin for custom operator npu_rotated_overlaps
 */

#include "onnx_common.h"
#include "op_cv_proto_extend.h"

using namespace std;
using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuRotatedOverlaps(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int input_size = node->input_size();
    int output_size = node->output_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    bool trans = false;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "trans" && attr.type() == ge::onnx::AttributeProto::INT) {
            trans = (attr.i() == 1);
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("trans", trans);
    op_dest.SetAttr("original_type", "npu::1::NPURotatedOverlaps");
    return SUCCESS;
}

static Status ParseOpToGraphNpuRotatedOverlaps(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    bool trans;
    if (op.GetAttr("trans", trans) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr trans failed.");
        return FAILED;
    }

    std::vector<int32_t> perm_box1 = {0, 2, 1};
    auto tensor_perm_box1 = Vec2Tensor(perm_box1, {3}, ge::DT_INT32);
    auto const_perm_box1 = op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_perm_box1);
    auto transpose_op_0 = op::Transpose((ori_name + "_Transpose_0").c_str())
                              .set_input_x(data0)
                              .set_input_perm(const_perm_box1);

    std::vector<int32_t> perm_box2 = {0, 2, 1};
    auto tensor_perm_box2 = Vec2Tensor(perm_box2, {3}, ge::DT_INT32);
    auto const_perm_box2 = op::Const((ori_name + "_Const_1").c_str()).set_attr_value(tensor_perm_box2);
    auto transpose_op_1 = op::Transpose((ori_name + "_Transpose_1").c_str())
                              .set_input_x(data1)
                              .set_input_perm(const_perm_box2);

    auto rotated_npu_overlaps_op = op::RotatedOverlaps((ori_name + "_RotatedOverlaps").c_str())
                                       .set_input_boxes(transpose_op_0)
                                       .set_input_query_boxes(transpose_op_1)
                                       .set_attr_trans(trans);

    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(rotated_npu_overlaps_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_rotated_overlaps op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("npu::1::NPURotatedOverlaps"), ge::AscendString("ai.onnx::11::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::12::NPURotatedOverlaps"), ge::AscendString("ai.onnx::13::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::14::NPURotatedOverlaps"), ge::AscendString("ai.onnx::15::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::16::NPURotatedOverlaps"), ge::AscendString("ai.onnx::17::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::18::NPURotatedOverlaps")})
    .ParseParamsFn(ParseParamsNpuRotatedOverlaps)
    .ParseOpToGraphFn(ParseOpToGraphNpuRotatedOverlaps)
    .ImplyType(ImplyType::TVM);
} // namespace domi
