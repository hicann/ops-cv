/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "cv_plugin_util.h"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuIou(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::string mode_str = "iou";
    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "mode" && attr.contains("i") && attr["i"].get<int64_t>() == 1) {
                        mode_str = "iof";
                        break;
                    }
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }
    op_dest.SetAttr("mode", mode_str);
    op_dest.SetAttr("eps", 0.01f);
    op_dest.SetAttr("aligned", false);
    return SUCCESS;
}

// register npu_iou op info to GE
REGISTER_CUSTOM_OP("Iou")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUIou"), ge::AscendString("ai.onnx::11::NPUIou"),
                   ge::AscendString("ai.onnx::12::NPUIou"), ge::AscendString("ai.onnx::13::NPUIou"),
                   ge::AscendString("ai.onnx::14::NPUIou"), ge::AscendString("ai.onnx::15::NPUIou"),
                   ge::AscendString("ai.onnx::16::NPUIou"), ge::AscendString("ai.onnx::17::NPUIou"),
                   ge::AscendString("ai.onnx::18::NPUIou")})
    .ParseParamsByOperatorFn(ParseParamsNpuIou)
    .ImplyType(ImplyType::TVM);
} // namespace domi
