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
 * \file grid_sampler2d.cpp
 * \brief
 */
#include "grid_sampler2d.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GridSampler2D);
static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;
static const string INTERPOLATION_BILINEAR = "bilinear";
static const string INTERPOLATION_NEAREST = "nearest";
static const string INTERPOLATION_BICUBIC = "bicubic";

static const string PADDING_ZEROS = "zeros";
static const string PADDING_BORDER = "border";
static const string PADDING_REFLECTION = "reflection";

static const int64_t INTERPOLATION_MODE_MIN_VALUE = 0;
static const int64_t INTERPOLATION_MODE_MAX_VALUE = 2;
static const int64_t INTERPOLATION_MODE_BILINEAR_VALUE = 0;
static const int64_t INTERPOLATION_MODE_NEAREST_VALUE = 1;
static const int64_t INTERPOLATION_MODE_BICUBIC_VALUE = 2;
static const int64_t PADDING_MODE_MIN_VALUE = 0;
static const int64_t PADDING_MODE_MAX_VALUE = 2;

static bool CheckAttrValid(int64_t interpolationMode, int64_t paddingMode)
{
    // 检查interpolationMode 、paddingMode 、 schedulerMode 是否在支持范围内
    if (interpolationMode < INTERPOLATION_MODE_MIN_VALUE || interpolationMode > INTERPOLATION_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "interpolationMode %ld should be in range [%ld, %ld].",
            interpolationMode,
            INTERPOLATION_MODE_MIN_VALUE,
            INTERPOLATION_MODE_MAX_VALUE);
        return false;
    }

    if (paddingMode < PADDING_MODE_MIN_VALUE || paddingMode > PADDING_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "paddingMode %ld should be in range [%ld, %ld].",
            paddingMode,
            PADDING_MODE_MIN_VALUE,
            PADDING_MODE_MAX_VALUE);
        return false;
    }
    return true;
}

inline const string &GetInterpolationModeStr(int64_t interpolationModeVal)
{
    if (interpolationModeVal == 0) {
        return INTERPOLATION_BILINEAR;
    }
    if (interpolationModeVal == 1) {
        return INTERPOLATION_NEAREST;
    }
    return INTERPOLATION_BICUBIC;
}

inline const string &GetPaddingModeStr(int64_t paddingModeVal)
{
    if (paddingModeVal == 0) {
        return PADDING_ZEROS;
    }
    if (paddingModeVal == 1) {
        return PADDING_BORDER;
    }
    return PADDING_REFLECTION;
}

const aclTensor *GridSampler2D(const aclTensor *x, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, aclOpExecutor *executor)
{
    L0_DFX(GridSampler2D, x, grid, interpolationMode, paddingMode, alignCorners);
    op::Shape outShape;
    outShape.AppendDim(x->GetViewShape().GetDim(FIRST_DIM));
    // 'C' dim of output shape
    outShape.AppendDim(x->GetViewShape().GetDim(SECOND_DIM));
    outShape.AppendDim(grid->GetViewShape().GetDim(SECOND_DIM));
    outShape.AppendDim(grid->GetViewShape().GetDim(THIRD_DIM));

    auto y = executor->AllocTensor(outShape, x->GetDataType(), op::Format::FORMAT_ND);
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc y tensor failed.");
        return nullptr;
    }

    static internal::AicpuTaskSpace space("GridSampler2D", ge::DEPEND_IN_SHAPE, false);
    if (CheckAttrValid(interpolationMode, paddingMode)) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICPU(GridSampler2D,
            OP_ATTR_NAMES({"interpolation_mode", "padding_mode", "align_corners"}),
            OP_INPUT(x, grid),
            OP_OUTPUT(y),
            OP_ATTR(GetInterpolationModeStr(interpolationMode), GetPaddingModeStr(paddingMode), alignCorners));
        CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    } else {
        return nullptr;
    }
    return y;
}
}  // namespace l0op