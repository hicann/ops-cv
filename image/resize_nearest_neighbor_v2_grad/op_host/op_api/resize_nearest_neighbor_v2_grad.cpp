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
 * \file resize_nearest_neighbor_v2_grad.cpp
 * \brief resize_nearest_neighbor_v2_grad
 */

#include "resize_nearest_neighbor_v2_grad.h"
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
OP_TYPE_REGISTER(ResizeNearestNeighborV2Grad);

static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;
static constexpr size_t NCHW_DIM_H = 2;
static constexpr size_t NCHW_DIM_W = 3;
static constexpr size_t NHWC_DIM_H = 1;
static constexpr size_t NHWC_DIM_W = 2;

// AICORE算子kernel 仅支持float
static const aclTensor *ResizeNearestNeighborV2GradAICORE(const aclTensor *grads, const aclTensor *inputSize,
    bool alignCorners, bool halfPixelCenters, const aclFloatArray *scales, aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(ResizeNearestNeighborV2GradAICORE, grads, inputSize, alignCorners, halfPixelCenters, out);
    auto ret = ACLNN_SUCCESS;
    if (scales == nullptr) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeNearestNeighborV2Grad,
            OP_INPUT(grads, inputSize),
            OP_OUTPUT(out),
            OP_ATTR(alignCorners, halfPixelCenters));
    } else {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeNearestNeighborV2Grad,
            OP_INPUT(grads, inputSize),
            OP_OUTPUT(out),
            OP_ATTR(alignCorners, halfPixelCenters, scales));
    }
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeNearestNeighborV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor *ResizeNearestNeighborV2Grad5Hd(const aclTensor *grads, const aclIntArray *inputSize, bool alignCorners,
    bool halfPixelCenters, aclOpExecutor *executor)
{
    int64_t inputSizeH = (*inputSize)[NCHW_DIM_H];
    int64_t inputSizeW = (*inputSize)[NCHW_DIM_W];
    auto format = grads->GetOriginalFormat();
    if (format == Format::FORMAT_NHWC) {
        inputSizeH = (*inputSize)[NHWC_DIM_H];
        inputSizeW = (*inputSize)[NHWC_DIM_W];
    }
    const int64_t shapes[2] = {inputSizeH, inputSizeW};
    aclIntArray *shapeArray = executor->AllocIntArray(shapes, sizeof(shapes) / sizeof(int64_t));
    CHECK_RET(shapeArray != nullptr, nullptr);
    auto inputSizeTensor = executor->ConvertToTensor(shapeArray, op::ToOpDataType(ACL_INT32));
    CHECK_RET(inputSizeTensor != nullptr, nullptr);

    op::Shape gradsStorageShape = grads->GetStorageShape();
    op::Shape gradsOriginalShape = grads->GetOriginalShape();
    gradsStorageShape.SetDim(DIM_H, inputSizeH);
    gradsStorageShape.SetDim(DIM_W, inputSizeW);
    if (format == Format::FORMAT_NCHW) {
        gradsOriginalShape.SetDim(NCHW_DIM_H, inputSizeH);
        gradsOriginalShape.SetDim(NCHW_DIM_W, inputSizeW);
    } else {
        gradsOriginalShape.SetDim(NHWC_DIM_H, inputSizeH);
        gradsOriginalShape.SetDim(NHWC_DIM_W, inputSizeW);
    }

    auto out = executor->AllocTensor(gradsStorageShape,
        gradsOriginalShape,
        grads->GetDataType(),
        grads->GetStorageFormat(),
        grads->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);

    return ResizeNearestNeighborV2GradAICORE(
        grads, inputSizeTensor, alignCorners, halfPixelCenters, nullptr, out, executor);
}

const aclTensor *ResizeNearestNeighborV2Grad(const aclTensor *grads, const aclIntArray *inputSize, bool alignCorners,
    bool halfPixelCenters, const aclFloatArray *scales, aclOpExecutor *executor)
{
    int64_t inputSizeH = 0;
    int64_t inputSizeW = 0;
    if (grads->GetOriginalFormat() == op::Format::FORMAT_NCHW) {
        inputSizeH = (*inputSize)[NCHW_DIM_H];
        inputSizeW = (*inputSize)[NCHW_DIM_W];
    } else if (grads->GetOriginalFormat() == op::Format::FORMAT_NHWC) {
        inputSizeH = (*inputSize)[NHWC_DIM_H];
        inputSizeW = (*inputSize)[NHWC_DIM_W];
    }

    const int64_t shapes[2] = {inputSizeH, inputSizeW};
    aclIntArray *shapeArray = executor->AllocIntArray(shapes, sizeof(shapes) / sizeof(int64_t));
    CHECK_RET(shapeArray != nullptr, nullptr);
    auto inputSizeTensor = executor->ConvertToTensor(shapeArray, op::ToOpDataType(ACL_INT32));
    CHECK_RET(inputSizeTensor != nullptr, nullptr);

    op::Shape gradsStorageShape = grads->GetStorageShape();
    op::Shape gradsOriginalShape = grads->GetOriginalShape();
    if (grads->GetOriginalFormat() == op::Format::FORMAT_NCHW) {
        gradsStorageShape.SetDim(NCHW_DIM_H, inputSizeH);
        gradsStorageShape.SetDim(NCHW_DIM_W, inputSizeW);
        gradsOriginalShape.SetDim(NCHW_DIM_H, inputSizeH);
        gradsOriginalShape.SetDim(NCHW_DIM_W, inputSizeW);
    } else if (grads->GetOriginalFormat() == op::Format::FORMAT_NHWC) {
        gradsStorageShape.SetDim(NHWC_DIM_H, inputSizeH);
        gradsStorageShape.SetDim(NHWC_DIM_W, inputSizeW);
        gradsOriginalShape.SetDim(NHWC_DIM_H, inputSizeH);
        gradsOriginalShape.SetDim(NHWC_DIM_W, inputSizeW);
    }

    auto out = executor->AllocTensor(gradsStorageShape,
        gradsOriginalShape,
        grads->GetDataType(),
        grads->GetStorageFormat(),
        grads->GetOriginalFormat());

    CHECK_RET(out != nullptr, nullptr);

    return ResizeNearestNeighborV2GradAICORE(
        grads, inputSizeTensor, alignCorners, halfPixelCenters, scales, out, executor);
}
}  // namespace l0op