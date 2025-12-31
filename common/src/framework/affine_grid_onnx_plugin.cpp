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
 * \file affine_grid_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsAffineGrid(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    bool align_corners = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "align_corners" && attr.i() != 0) {
            align_corners = true;
            break;
        }
    }
    op_dest.SetAttr("align_corners", align_corners);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AffineGrid")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::AffineGrid"),
                 ge::AscendString("ai.onnx::9::AffineGrid"),
                 ge::AscendString("ai.onnx::10::AffineGrid"),
                 ge::AscendString("ai.onnx::11::AffineGrid"),
                 ge::AscendString("ai.onnx::12::AffineGrid"),
                 ge::AscendString("ai.onnx::13::AffineGrid"),
                 ge::AscendString("ai.onnx::14::AffineGrid"),
                 ge::AscendString("ai.onnx::15::AffineGrid"),
                 ge::AscendString("ai.onnx::16::AffineGrid")})
  .ParseParamsFn(ParseParamsAffineGrid)
  .ImplyType(ImplyType::TVM);
} // namespace domi
