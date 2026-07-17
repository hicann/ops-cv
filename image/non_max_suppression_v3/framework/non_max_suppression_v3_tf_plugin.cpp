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
#include "log/log.h"
#include <map>

namespace domi {
static Status NonMaxSuppressionV3MappingFn(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE("NonMaxSuppressionV3MappingFn", "op[NonMaxSuppressionV3] tf plugin parser[AutoMappingFn] failed.");
        return FAILED;
    }

    op.SetAttr("offset", 0);
    OP_LOGI("NonMaxSuppressionV3MappingFn", "op[NonMaxSuppressionV3] tf plugin parser finish.");
    return SUCCESS;
}

// register NonMaxSuppressionV3 op to GE
REGISTER_CUSTOM_OP("NonMaxSuppressionV3")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV3")
    .ParseParamsByOperatorFn(NonMaxSuppressionV3MappingFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
