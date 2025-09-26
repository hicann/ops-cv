/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_bilinear_v2.cpp
 * \brief resize_bilinear_v2
 */
#include "resize_bilinear_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ResizeBilinearV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910_95 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self)
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910_95);
    }

    // ResizeBilinearV2只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor *ResizeBilinearV2AICPU(
    const aclTensor *x, const aclTensor *size, const bool align_corners, const aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2AICPU, x, size, align_corners, y);
    const bool half_pixel_centers = !align_corners;

    // tf ResizeBilinear only support float output and format NHWC
    aclTensor *out = executor->AllocTensor(y->GetViewShape(), op::DataType::DT_FLOAT, y->GetViewFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    static internal::AicpuTaskSpace space("ResizeBilinear", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ResizeBilinearV2,
        OP_ATTR_NAMES({"T", "align_corners", "half_pixel_centers"}),
        OP_INPUT(x, size),
        OP_OUTPUT(out),
        OP_ATTR(x->GetDataType(), align_corners, half_pixel_centers));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

static const aclTensor *ResizeBilinearV2AICORE(
    const aclTensor *x, const aclTensor *size, const bool align_corners, const aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2AICORE, x, size, align_corners, y);
    const bool half_pixel_centers = !align_corners;
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeBilinearV2, OP_INPUT(x, size), OP_OUTPUT(y), OP_ATTR(align_corners, half_pixel_centers));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeBilinearV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}
const aclTensor *ResizeBilinearV2(
    const aclTensor *x, const aclTensor *size, const bool align_corners, const aclTensor *y, aclOpExecutor *executor)
{
    if (IsAiCoreSupport(x)) {
        return ResizeBilinearV2AICORE(x, size, align_corners, y, executor);
    } else {
        return ResizeBilinearV2AICPU(x, size, align_corners, y, executor);
    }
}

static const aclTensor *ResizeBilinearV2AicoreWith4d(const aclTensor *x, const aclTensor *size,
    const bool align_corners, const aclFloatArray *scales, const aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2AicoreWith4d, x, size, align_corners, scales, y);
    const bool half_pixel_centers = !align_corners;

    auto ret = ACLNN_SUCCESS;
    if (scales == nullptr) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeBilinearV2,
            OP_INPUT(x, size),
            OP_OUTPUT(y),
            OP_ATTR(align_corners, half_pixel_centers, y->GetDataType()));
    } else {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeBilinearV2,
            OP_INPUT(x, size),
            OP_OUTPUT(y),
            OP_ATTR(align_corners, half_pixel_centers, y->GetDataType(), scales));
    }
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ADD_TO_LAUNCHER_LIST_AICORE with scales failed."),
        return nullptr);
    return y;
}

const aclTensor *ResizeBilinearV2With4d(const aclTensor *x, const aclTensor *size, const bool align_corners,
    const aclFloatArray *scales, const aclTensor *y, aclOpExecutor *executor)
{
    if (IsAiCoreSupport(x)) {
        return ResizeBilinearV2AicoreWith4d(x, size, align_corners, scales, y, executor);
    } else {
        return ResizeBilinearV2AICPU(x, size, align_corners, y, executor);
    }
}

}  // namespace l0op
