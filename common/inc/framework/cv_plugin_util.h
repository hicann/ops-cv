/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CV_COMMON_CV_PLUGIN_UTIL_H
#define CV_COMMON_CV_PLUGIN_UTIL_H

#include <string>
#include <vector>

#include "graph/operator.h"
#include "log/log.h"
#include "register/register.h"

namespace domi {
inline Status StrToFloat(const std::string& value, float& result)
{
    try {
        size_t pos = 0;
        result = std::stof(value, &pos);
        if (pos != value.size()) {
            return FAILED;
        }
    } catch (...) {
        return FAILED;
    }
    return SUCCESS;
}

template <typename T>
inline std::string GetOpName(const T& op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}

inline Status ChangeFormatFromOnnx(ge::Operator& op, const int idx, ge::Format format, bool is_input)
{
    if (is_input) {
        ge::TensorDesc org_tensor = op.GetInputDesc(idx);
        org_tensor.SetOriginFormat(format);
        org_tensor.SetFormat(format);
        auto ret = op.UpdateInputDesc(idx, org_tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change input format failed.");
            return FAILED;
        }
    } else {
        ge::TensorDesc org_tensor_y = op.GetOutputDesc(idx);
        org_tensor_y.SetOriginFormat(format);
        org_tensor_y.SetFormat(format);
        auto ret_y = op.UpdateOutputDesc(idx, org_tensor_y);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace domi

#endif // CV_COMMON_CV_PLUGIN_UTIL_H
