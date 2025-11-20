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
 * \file upsample_nearest_3d.cpp
 * \brief
 */
#include "upsample_nearest_3d.h"
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
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearest3d);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND310P_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

const aclTensor *UpsampleNearest3dNcdhw(const aclTensor *self, const aclIntArray *outputSize,
    const aclFloatArray *scales, const aclFloatArray *castScales, aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearest3dNcdhw, self, outputSize, scales);

    // 获取DHW维度Size D = outputSize[0], H = outputSize[1], W = outputSize[2]
    const int64_t sizeD = (*outputSize)[DIM_ZERO];
    const int64_t sizeH = (*outputSize)[DIM_ONE];
    const int64_t sizeW = (*outputSize)[DIM_TWO];
    float scales_d = 0.0;
    float scales_h = 0.0;
    float scales_w = 0.0;
    if (castScales->Size() == DIM_THREE) {
        scales_d = (*castScales)[DIM_ZERO];
        scales_h = (*castScales)[DIM_ONE];
        scales_w = (*castScales)[DIM_TWO];
    }
    // 生成out shape为 (N，C，outputSize[0], outputSize[1], outputSize[2])
    op::Shape selfStorageShape = self->GetStorageShape();
    op::Shape selfOriginalShape = self->GetOriginalShape();
    selfStorageShape.SetDim(DIM_TWO, sizeD);
    selfStorageShape.SetDim(DIM_THREE, sizeH);
    selfStorageShape.SetDim(DIM_FOUR, sizeW);
    selfOriginalShape.SetDim(DIM_TWO, sizeD);
    selfOriginalShape.SetDim(DIM_THREE, sizeH);
    selfOriginalShape.SetDim(DIM_FOUR, sizeW);

    const aclTensor *out = executor->AllocTensor(
        selfStorageShape, selfOriginalShape, self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    if ((socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) &&
        CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        ADD_TO_LAUNCHER_LIST_AICORE(
            UpsampleNearest3d, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize, scales_d, scales_h, scales_w));
        return out;
    }
    if ((socVer == SocVersion::ASCEND310P) && CheckType(self->GetDataType(), ASCEND310P_AICORE_DTYPE_SUPPORT_LIST)) {
        ADD_TO_LAUNCHER_LIST_AICORE(
            UpsampleNearest3d, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize, scales_d, scales_h, scales_w));
        return out;
    }

    static internal::AicpuTaskSpace space("UpsampleNearest3d");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UpsampleNearest3d,
        OP_ATTR_NAMES({"output_size", "scales"}),
        OP_INPUT(self),
        OP_OUTPUT(out),
        OP_ATTR(outputSize, scales));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}
}  // namespace l0op
