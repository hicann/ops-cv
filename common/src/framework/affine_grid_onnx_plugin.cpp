/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "cv_plugin_util.h"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsAffineGrid(const ge::Operator& op_src, ge::Operator& op_dest)
{
    bool align_corners = false;
    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "align_corners" && attr.contains("i")) {
                        align_corners = attr["i"].get<int64_t>() != 0;
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
    op_dest.SetAttr("align_corners", align_corners);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AffineGrid")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AffineGrid"), ge::AscendString("ai.onnx::9::AffineGrid"),
                   ge::AscendString("ai.onnx::10::AffineGrid"), ge::AscendString("ai.onnx::11::AffineGrid"),
                   ge::AscendString("ai.onnx::12::AffineGrid"), ge::AscendString("ai.onnx::13::AffineGrid"),
                   ge::AscendString("ai.onnx::14::AffineGrid"), ge::AscendString("ai.onnx::15::AffineGrid"),
                   ge::AscendString("ai.onnx::16::AffineGrid")})
    .ParseParamsByOperatorFn(ParseParamsAffineGrid)
    .ImplyType(ImplyType::TVM);
} // namespace domi
