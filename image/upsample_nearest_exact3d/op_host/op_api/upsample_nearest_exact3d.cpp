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
 * \file upsample_nearest_exact3d.cpp
 * \brief
 */
#include "upsample_nearest_exact3d.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearestExact3d);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

const aclTensor *UpsampleNearestExact3dNcdhw(
    const aclTensor *self, const aclIntArray *outputSize, const aclFloatArray *scales, aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearestExact3dNcdhw, self, outputSize, scales);

    // 获取DHW维度Size D = outputSize[0], H = outputSize[1], W = outputSize[2]
    const int64_t sizeD = (*outputSize)[DIM_ZERO];
    const int64_t sizeH = (*outputSize)[DIM_ONE];
    const int64_t sizeW = (*outputSize)[DIM_TWO];
    float scales_d = 0.0;
    float scales_h = 0.0;
    float scales_w = 0.0;
    if (scales->Size() == DIM_THREE) {
        scales_d = (*scales)[DIM_ZERO];
        scales_h = (*scales)[DIM_ONE];
        scales_w = (*scales)[DIM_TWO];
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

    ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleNearestExact3d, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize, scales_d, scales_h, scales_w));
    return out;
}
}  // namespace l0op
