/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"
#include "cv_plugin_util.h"

namespace domi {
static Status ToAbsoluteBBoxParserParams(const ge::Operator& op_src, ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter ToAbsoluteBBox fusion parser.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ToAbsoluteBBox")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ToAbsoluteBBox")
    .ParseParamsByOperatorFn(ToAbsoluteBBoxParserParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
