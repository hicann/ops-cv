/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_iou.h"
#include "iou_v2.h"
#include <dlfcn.h>
#include <acl/acl.h>
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C"
{
#endif

  static const int64_t TENSOR_DIM_NUM = 2;
  static const int64_t INPUT_SECOND_DIM = 4;

  static const std::initializer_list<op::DataType> TYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

  static bool CheckNotNull(const aclTensor *bBoxes, const aclTensor *gtBoxes, const aclTensor *overlap)
  {
    OP_CHECK_NULL(bBoxes, return false);
    OP_CHECK_NULL(gtBoxes, return false);
    OP_CHECK_NULL(overlap, return false);
    return true;
  }

  static bool CheckDtypeValid(const aclTensor *bBoxes, const aclTensor *gtBoxes, const aclTensor *overlap)
  {
    OP_CHECK_DTYPE_NOT_MATCH(bBoxes, gtBoxes->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(bBoxes, overlap->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(bBoxes, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gtBoxes, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(overlap, TYPE_SUPPORT_LIST, return false);

    if (bBoxes->GetStorageFormat() != Format::FORMAT_ND || gtBoxes->GetStorageFormat() != Format::FORMAT_ND ||
        overlap->GetStorageFormat() != Format::FORMAT_ND)
    {
      OP_LOGI("input and output format should be ND.");
      return false;
    }
    return true;
  }

  static bool CheckAttr(const char *mode, float eps)
  {
    // 检查self和other能否做数据类型推导
    if (strcmp(mode, "iou") != 0 && strcmp(mode, "iof") != 0)
    {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode should be [iou] or [iof], but get [%s].", mode);
      return false;
    }
    if (eps < 0)
    {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "eps should be greater than or equal to 0, but now is [%f].", eps);
      return false;
    }
    return true;
  }

  static bool CheckShape(const aclTensor *bBoxes, const aclTensor *gtBoxes, const aclTensor *overlap, bool aligned)
  {
    OP_CHECK(bBoxes->GetViewShape().GetDimNum() == TENSOR_DIM_NUM &&
                 gtBoxes->GetViewShape().GetDimNum() == TENSOR_DIM_NUM &&
                 overlap->GetViewShape().GetDimNum() == TENSOR_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor should be 2D."), return false);
    if (bBoxes->GetViewShape().GetDim(1) != INPUT_SECOND_DIM || gtBoxes->GetViewShape().GetDim(1) != INPUT_SECOND_DIM)
    {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "iou input shape dim1 should be 4.");
      return false;
    }
    if (aligned)
    {
      OP_CHECK(bBoxes->GetViewShape().GetDim(0) == gtBoxes->GetViewShape().GetDim(0),
               OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bBoxes shape dim0 and gtBoxes shape dim0 should be equal when aligned."),
               return false);
      OP_CHECK(overlap->GetViewShape().GetDim(1) == 1,
               OP_LOGE(ACLNN_ERR_PARAM_INVALID, "overlap shape dim1 should be 1 when aligned."),
               return false);
    }
    return true;
  }

  static aclnnStatus CheckParams(const aclTensor *bBoxes, const aclTensor *gtBoxes,
                                 aclTensor *overlap, const char *mode, float eps,
                                 bool aligned)
  {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(bBoxes, gtBoxes, overlap), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(bBoxes, gtBoxes, overlap), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性
    CHECK_RET(CheckAttr(mode, eps), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(bBoxes, gtBoxes, overlap, aligned), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
  }

  aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor *bBoxes, const aclTensor *gtBoxes,
                                       const char *mode, float eps, bool aligned,
                                       aclTensor *overlap, uint64_t *workspaceSize, aclOpExecutor **executor)
  {
    L2_DFX_PHASE_1(aclnnIou, DFX_IN(bBoxes, gtBoxes, mode, eps, aligned), DFX_OUT(overlap));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(bBoxes, gtBoxes, overlap, mode, eps, aligned);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (bBoxes->IsEmpty() || gtBoxes->IsEmpty())
    {
      OP_LOGD("aclnnIou, dealing with empty tensor.");
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
    }

    // 固定写法，将输入bBoxes转换成连续的tensor
    auto bBoxesContiguous = l0op::Contiguous(bBoxes, uniqueExecutor.get());
    CHECK_RET(bBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入gtBoxes转换成连续的tensor
    auto gtBoxesContiguous = l0op::Contiguous(gtBoxes, uniqueExecutor.get());
    CHECK_RET(gtBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t valuePerm[2] = {1, 0};                                 // 交换0轴和1轴
    auto perm = uniqueExecutor.get()->AllocIntArray(valuePerm, 2); // dims = 2
    // aligned为true时，输入进行转置
    if (aligned)
    {
      bBoxesContiguous = l0op::Transpose(bBoxesContiguous, perm, uniqueExecutor.get());
      CHECK_RET(bBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
      gtBoxesContiguous = l0op::Transpose(gtBoxesContiguous, perm, uniqueExecutor.get());
      CHECK_RET(gtBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 进行计算
    auto iouOut = l0op::IouV2(bBoxesContiguous, gtBoxesContiguous, mode, eps, aligned, uniqueExecutor.get());
    CHECK_RET(iouOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // aligned为false时，输出进行转置
    if (!aligned)
    {
      auto outTransData = l0op::Transpose(iouOut, perm, uniqueExecutor.get());
      CHECK_RET(outTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);
      // 固定写法，将计算结果拷贝到输出overlap上，overlap可能是非连续的tensor
      auto viewCopyResult = l0op::ViewCopy(outTransData, overlap, uniqueExecutor.get());
      CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    else
    {
      // 固定写法，将计算结果拷贝到输出overlap上，overlap可能是非连续的tensor
      auto viewCopyResult = l0op::ViewCopy(iouOut, overlap, uniqueExecutor.get());
      CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
  }

  aclnnStatus aclnnIou(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
  {
    L2_DFX_PHASE_2(aclnnIou);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
  }

#ifdef __cplusplus
}
#endif