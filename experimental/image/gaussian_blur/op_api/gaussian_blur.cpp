/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gaussian_blur.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GaussianBlur);

static const std::initializer_list<DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static inline bool IsAiCoreSupport(const aclTensor* self)
{
    OP_CHECK(GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur only supports Ascend950."), return false);
    OP_CHECK(CheckType(self->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur does not support dtype %s.",
                     op::ToString(self->GetDataType()).GetString()),
             return false);
    return true;
}

static const aclTensor* LaunchGaussianBlur(const aclTensor* src, const aclIntArray* ksize, float sigmaX, float sigmaY,
                                           int64_t borderType, aclTensor* dst, aclOpExecutor* executor)
{
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GaussianBlur, OP_INPUT(src), OP_OUTPUT(dst),
                                           OP_ATTR(ksize, sigmaX, sigmaY, borderType));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GaussianBlur ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return dst;
}

const aclTensor* GaussianBlur(const aclTensor* src, const aclIntArray* ksize, float sigmaX, float sigmaY,
                              int64_t borderType, aclTensor* dst, aclOpExecutor* executor)
{
    L0_DFX(GaussianBlur, src, ksize, sigmaX, sigmaY, borderType, dst);
    OP_CHECK(IsAiCoreSupport(src), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur support check failed."),
             return nullptr);
    return LaunchGaussianBlur(src, ksize, sigmaX, sigmaY, borderType, dst, executor);
}

} // namespace l0op
