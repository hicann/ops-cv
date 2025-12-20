/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_three_interpolate_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "three_interpolate_backward.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr uint8_t DIM_INDEX_B = 0;
constexpr uint8_t DIM_INDEX_C = 1;
constexpr uint8_t DIM_INDEX_N = 2;
constexpr uint8_t DIM_INDEX_M = 2;
constexpr uint8_t NEW_SHAPE_DIMS = 4;
constexpr uint8_t CONST_1 = 1;
constexpr uint8_t CONST_2 = 2;
constexpr uint8_t CONST_3 = 3;
} // namespace

static bool CheckNotNull(
    const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, const aclTensor* grad_y)
{
    OP_CHECK_NULL(grad_x, return false);
    OP_CHECK_NULL(idx, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(grad_y, return false);
    return true;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> IDX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static bool CheckDtypeValid(const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight)
{
    // 检查gradOutput和self数据类型是否在ThresholdBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(grad_x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, const aclTensor* grad_y)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(grad_x, idx, weight, grad_y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否正确
    CHECK_RET(CheckDtypeValid(grad_x, idx, weight), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static void CreateGradXReshapeData(const aclTensor* grad_x, FVector<int64_t>& newShapeVector)
{
    const auto& shape = grad_x->GetViewShape();
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_B));
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_C));
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_M));
}

aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(
    const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, int m, aclTensor* grad_y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnThreeInterpolateBackward, DFX_IN(grad_x, idx, weight), DFX_OUT(grad_y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(grad_x, idx, weight, grad_y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (grad_x->IsEmpty() || idx->IsEmpty() || weight->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto grad_x_contiguous = l0op::Contiguous(grad_x, uniqueExecutor.get());
    CHECK_RET(grad_x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx_contiguous = l0op::Contiguous(idx, uniqueExecutor.get());
    CHECK_RET(grad_x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weight_contiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weight_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // grad_x:(b,c,n,1)(NCHW) -> (b,c1,n,1,c0)(5HD)
    auto grad_x_5d = l0op::TransDataSpecial(grad_x_contiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
    CHECK_RET(grad_x_5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // grad_x:(b,c1,n,1,c0)(5HD) -> grad_y:(b,c1,m,1,c0)(5HD)
    auto out_5d = l0op::ThreeInterpolateBackward(grad_x_5d, idx_contiguous, weight_contiguous, m, uniqueExecutor.get());
    CHECK_RET(out_5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out_4d = l0op::TransDataSpecial(out_5d, op::Format::FORMAT_NCHW, 0, uniqueExecutor.get());
    CHECK_RET(out_4d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(out_4d, grad_y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThreeInterpolateBackward(
    void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThreeInterpolateBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif