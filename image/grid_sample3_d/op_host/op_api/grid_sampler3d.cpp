/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grid_sampler3d.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GridSampler3D);

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;
static const size_t FIFTH_DIM = 4;

static const string INTERPOLATION_BILINEAR = "bilinear";
static const string INTERPOLATION_NEAREST = "nearest";

static const string PADDING_ZEROS = "zeros";
static const string PADDING_BORDER = "border";
static const string PADDING_REFLECTION = "reflection";

inline const string &GetInterpolationModeStr(int64_t interpolationMode)
{
    if (interpolationMode == 0) {
        return INTERPOLATION_BILINEAR;
    }
    return INTERPOLATION_NEAREST;
}

inline const string &GetPaddingModeStr(int64_t paddingMode)
{
    if (paddingMode == 0) {
        return PADDING_ZEROS;
    }
    if (paddingMode == 1) {
        return PADDING_BORDER;
    }
    return PADDING_REFLECTION;
}

const aclTensor *GridSampler3D(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, aclOpExecutor *executor)
{
    L0_DFX(GridSampler3D, input, grid, interpolationMode, paddingMode, alignCorners);
    op::Shape outShape;
    const op::Format inputFormat = input->GetStorageFormat();
    if (inputFormat == op::Format::FORMAT_NDHWC) {
        outShape.AppendDim(input->GetViewShape().GetDim(FIRST_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(SECOND_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(THIRD_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(FOURTH_DIM));
        outShape.AppendDim(input->GetViewShape().GetDim(FIFTH_DIM));
    } else if ((inputFormat == op::Format::FORMAT_NCDHW) || (inputFormat == op::Format::FORMAT_ND)) {
        outShape.AppendDim(input->GetViewShape().GetDim(FIRST_DIM));
        outShape.AppendDim(input->GetViewShape().GetDim(SECOND_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(SECOND_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(THIRD_DIM));
        outShape.AppendDim(grid->GetViewShape().GetDim(FOURTH_DIM));
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported input format %s", op::ToString(inputFormat).GetString());
        return nullptr;
    }

    auto out = executor->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    const auto data_format_str =
        inputFormat == op::Format::FORMAT_ND ? op::ToString(op::Format::FORMAT_NCDHW) : op::ToString(inputFormat);
    static internal::AicpuTaskSpace space("GridSampler3D", ge::DEPEND_IN_SHAPE, false);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(GridSampler3D,
        OP_ATTR_NAMES({"interpolation_mode", "padding_mode", "data_format", "align_corners"}),
        OP_INPUT(input, grid),
        OP_OUTPUT(out),
        OP_ATTR(GetInterpolationModeStr(interpolationMode),
            GetPaddingModeStr(paddingMode),
            data_format_str.GetString(),
            alignCorners));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GridSampler3D AiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}
}  // namespace l0op
