/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_non_max_suppression.h"
#include "non_max_suppression_v6.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static constexpr size_t DIM_TWO = 2;
static constexpr size_t NUM_FOUR = 4;
static constexpr int32_t MAX_VALID_OUTPUT = 700;

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor *boxes, const aclTensor *scores, aclFloatArray *iouThreshold, aclTensor *selectedIndices) {
    OP_CHECK_NULL(boxes, return false);
    OP_CHECK_NULL(scores, return false);
    if (iouThreshold->Size() <= 0) {
        return false;
    }
    OP_CHECK_NULL(selectedIndices, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *boxes, const aclTensor *scores, aclFloatArray *iouThreshold) {
    // 检查输入的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(boxes, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scores, FLOAT_DTYPE_SUPPORT_LIST, return false);
    // 检查boxes和scores数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(boxes, scores, return false);
    if ((iouThreshold->operator[](0) > 1) || (iouThreshold->operator[](0) < 0)) {
        return false;
    }
    return true;
}

static bool CheckFormatValid(const aclTensor *boxes, const aclTensor *scores, aclTensor *selectedIndices)
{
    return boxes->GetStorageFormat() == op::Format::FORMAT_ND && scores->GetStorageFormat() == op::Format::FORMAT_ND && selectedIndices->GetStorageFormat() == op::Format::FORMAT_ND;
}

static bool CheckShape(const aclTensor* boxes, const aclTensor *scores) {
    OP_CHECK_WRONG_DIMENSION(boxes, 3, return false);
    OP_CHECK_WRONG_DIMENSION(scores, 3, return false);

    auto const &boxesShape = boxes->GetViewShape();
    auto const &scoresShape = scores->GetViewShape();
    if (boxesShape.GetDim(0) != scoresShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim0 [%ld] and scores shape dim0 [%ld] should be same", boxesShape.GetDim(0), scoresShape.GetDim(0));
        return false;
    }
    if (boxesShape.GetDim(1) != scoresShape.GetDim(DIM_TWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] and scores shape dim2 [%ld] should be same", boxesShape.GetDim(1), scoresShape.GetDim(DIM_TWO));
        return false;
    }
    if (boxesShape.GetDim(DIM_TWO) != NUM_FOUR) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] should be 4", boxesShape.GetDim(DIM_TWO));
        return false;
    }
    return true;
}

static bool CheckAttr(const int centerPointBox) {
    if (centerPointBox != 0 && centerPointBox != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "centerPointBox [%d] should be equal to 0 or 1", centerPointBox);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *boxes, const aclTensor *scores, aclFloatArray *iouThreshold, aclTensor *selectedIndices, int centerPointBox) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(boxes, scores, iouThreshold, selectedIndices), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内、且满足约束，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(boxes, scores, iouThreshold), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据格式是否在API支持范围之内
    CHECK_RET(CheckFormatValid(boxes, scores, selectedIndices), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(boxes, scores), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查属性数据是否合法
    CHECK_RET(CheckAttr(centerPointBox), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, aclIntArray *maxOutputBoxesPerClass,
    aclFloatArray *iouThreshold, aclFloatArray *scoreThreshold, int centerPointBox, aclTensor *selectedIndices, uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    int64_t maxBoxesSize = 0;
    L2_DFX_PHASE_1(aclnnNonMaxSuppression, DFX_IN(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox, maxBoxesSize), DFX_OUT(selectedIndices));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t maxOutputSize = 0;
    if (maxOutputBoxesPerClass->Size() > 0) {
        maxOutputSize = maxOutputBoxesPerClass->operator[](0);
    }
    if (maxOutputSize > MAX_VALID_OUTPUT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxOutputBoxesPerClass[%ld] should < 700 ", maxOutputSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret = CheckParams(boxes, scores, iouThreshold, selectedIndices, centerPointBox);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2002) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclnnNonMaxSuppression is not supported %u npuArch", static_cast<uint32_t>(curArch));
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    maxBoxesSize = maxOutputSize * scores->GetViewShape().GetDim(0) * scores->GetViewShape().GetDim(1);
    auto boxesContigous = l0op::Contiguous(boxes, uniqueExecutor.get());
    auto scoresContigous = l0op::Contiguous(scores, uniqueExecutor.get());

    // // 调用non_max_suppression算子
    const aclTensor *nmsOut = l0op::NonMaxSuppressionV6(boxesContigous, scoresContigous, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox, maxBoxesSize, selectedIndices, uniqueExecutor.get());

    // 固定写法，将计算结果拷贝到输出selectedIndices上，selectedIndices可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(nmsOut, selectedIndices, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNonMaxSuppression(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnNonMaxSuppression);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
