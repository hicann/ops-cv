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
 * \file aclnn_im2col_backward.cpp
 * \brief
 */
#include "aclnn_im2col_backward.h"
#include "col2im.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NEED_SQUEEZE = 2;
static constexpr size_t NO_NEED_SQUEEZE = 3;
static constexpr size_t ARRAY_SIZE = 2;
static constexpr size_t GRAD_DIM = 4;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor *gradOutput, const aclIntArray *inputSize,
    const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride,
    const aclTensor *out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtype(const aclTensor *gradOutput, const aclTensor *out)
{
    bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                             GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *out)
{
    if (gradOutput->GetViewShape().GetDimNum() != NEED_SQUEEZE &&
        gradOutput->GetViewShape().GetDimNum() != NO_NEED_SQUEEZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected gradOutput dim [%zu] to be 2 or 3 but check failed.",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }
    if ((gradOutput->GetViewShape().GetDimNum() == NEED_SQUEEZE &&
            out->GetViewShape().GetDimNum() != NO_NEED_SQUEEZE) ||
        (gradOutput->GetViewShape().GetDimNum() == NO_NEED_SQUEEZE && out->GetViewShape().GetDimNum() != GRAD_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected out dim [%zu] to be 1 greater than gradOutput dim [%zu] "
            "but check failed.",
            out->GetViewShape().GetDimNum(),
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static bool CheckArrayValue(
    const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride)
{
    OP_CHECK((*kernelSize)[0] > 0 && (*kernelSize)[1] > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Kernel size must be greater than zero, but got kernelSize=(%ld,%ld).",
            (*kernelSize)[0],
            (*kernelSize)[1]),
        return false);
    OP_CHECK((*stride)[0] > 0 && (*stride)[1] > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Stride must be greater than zero, but got stride=(%ld,%ld).",
            (*stride)[0],
            (*stride)[1]),
        return false);
    OP_CHECK((*dilation)[0] > 0 && (*dilation)[1] > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Dilation must be greater than zero, but got dilation=(%ld,%ld).",
            (*dilation)[0],
            (*dilation)[1]),
        return false);
    OP_CHECK((*padding)[0] >= 0 && (*padding)[1] >= 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Padding must not be negative, but got padding=(%ld,%ld).",
            (*padding)[0],
            (*padding)[1]),
        return false);
    return true;
}

static bool CheckArray(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclIntArray *kernelSize,
    const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride)
{
    if (inputSize->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected inputSize equals to 2, but got size %lu.", inputSize->Size());
        return false;
    }
    if (kernelSize->Size() != ARRAY_SIZE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected kernelSize equals to 2, but got size %lu.", kernelSize->Size());
        return false;
    }
    if (dilation->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected dilation equals to 2, but got size %lu.", dilation->Size());
        return false;
    }
    if (padding->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected padding equals to 2, but got size %lu.", padding->Size());
        return false;
    }
    if (stride->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected stride equals to 2, but got size %lu.", stride->Size());
        return false;
    }
    OP_CHECK(CheckArrayValue(kernelSize, dilation, padding, stride),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckArrayValue faild."),
        return false);
    size_t inputPlane = gradOutput->GetViewShape().GetDimNum() == NO_NEED_SQUEEZE
                            ? gradOutput->GetViewShape().GetDim(1)
                            : gradOutput->GetViewShape().GetDim(0);
    if (inputPlane % ((*kernelSize)[0] * (*kernelSize)[1]) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected size of gradOutput's dim -2 to be divisible by the product of kernelSize"
            " but got gradOutput.shape[-2]=%zu and kernelSize=(%ld,%ld).",
            inputPlane,
            (*kernelSize)[0],
            (*kernelSize)[1]);
        return false;
    }
    size_t numL = gradOutput->GetViewShape().GetDimNum() == NO_NEED_SQUEEZE ? gradOutput->GetViewShape().GetDim(2)
                                                                            : gradOutput->GetViewShape().GetDim(1);
    size_t numL0 = ((*inputSize)[0] + (*padding)[0] * 2 - (*dilation)[0] * ((*kernelSize)[0] - 1) - 1 + (*stride)[0]) /
                   (*stride)[0];
    size_t numL1 = ((*inputSize)[1] + (*padding)[1] * 2 - (*dilation)[1] * ((*kernelSize)[1] - 1) - 1 + (*stride)[1]) /
                   (*stride)[1];
    size_t numL0L1 = numL0 * numL1;
    if (numL != numL0L1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected gradOutput.shape[-1] should be %zu, but current is %zu.", numL0L1, numL);
        return false;
    }
    return true;
}

static bool CheckOutShape(
    const aclTensor *gradOutput, const aclIntArray *inputSize, const aclIntArray *kernelSize, const aclTensor *out)
{
    op::Shape im2colShape;
    if (gradOutput->GetViewShape().GetDimNum() == NEED_SQUEEZE) {
        size_t gradOutShape0 = gradOutput->GetViewShape().GetDim(0);
        im2colShape.AppendDim(gradOutShape0 / ((*kernelSize)[0] * (*kernelSize)[1]));
    } else {
        im2colShape.AppendDim(gradOutput->GetViewShape().GetDim(0));
        size_t gradOutShape1 = gradOutput->GetViewShape().GetDim(1);
        im2colShape.AppendDim(gradOutShape1 / ((*kernelSize)[0] * (*kernelSize)[1]));
    }
    im2colShape.AppendDim((*inputSize)[0]);
    im2colShape.AppendDim((*inputSize)[1]);
    auto outShape = out->GetViewShape();
    if (outShape != im2colShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Shape of out should be %s, but current is %s.",
            op::ToString(im2colShape).GetString(),
            op::ToString(outShape).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclIntArray *kernelSize,
    const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, inputSize, kernelSize, dilation, padding, stride, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtype(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape
    CHECK_RET(CheckShape(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数组是否满足要求
    CHECK_RET(CheckArray(gradOutput, inputSize, kernelSize, dilation, padding, stride), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查gradOutput和out的shape是否满足要求
    CHECK_RET(CheckOutShape(gradOutput, inputSize, kernelSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *inputSize,
    const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnIm2colBackward, DFX_IN(gradOutput, inputSize, kernelSize, dilation, padding, stride), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, inputSize, kernelSize, dilation, padding, stride, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool isNeedSqueeze = (gradOutput->GetViewShape().GetDimNum() == NEED_SQUEEZE);

    // 固定写法，将输入转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueeze =
        isNeedSqueeze ? l0op::UnsqueezeNd(gradOutputContiguous, static_cast<int64_t>(0), uniqueExecutor.get())
                      : gradOutputContiguous;
    CHECK_RET(gradOutputUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t newShape[] = {gradOutputUnsqueeze->GetViewShape().GetDim(0),
        gradOutputUnsqueeze->GetViewShape().GetDim(1) / ((*kernelSize)[0] * (*kernelSize)[1]),
        (*kernelSize)[0] * (*kernelSize)[1],
        gradOutputUnsqueeze->GetViewShape().GetDim(2)};

    auto gradOutputReshape = l0op::Reshape(
        gradOutputUnsqueeze, uniqueExecutor.get()->AllocIntArray(newShape, GRAD_DIM), uniqueExecutor.get());
    CHECK_RET(gradOutputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputReFormat = l0op::ReFormat(gradOutputReshape, op::Format::FORMAT_NCHW);
    CHECK_RET(gradOutputReFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputTransData =
        l0op::TransDataSpecial(gradOutputReFormat, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
    CHECK_RET(gradOutputTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto col2imOut =
        l0op::Col2im(gradOutputTransData, inputSize, kernelSize, dilation, padding, stride, uniqueExecutor.get());
    CHECK_RET(col2imOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outTransData = l0op::TransDataSpecial(col2imOut, op::Format::FORMAT_NCHW, 0, uniqueExecutor.get());
    CHECK_RET(outTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outSqueeze =
        isNeedSqueeze ? l0op::SqueezeNd(outTransData, static_cast<int64_t>(0), uniqueExecutor.get()) : outTransData;
    CHECK_RET(outSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outView =
        uniqueExecutor.get()->CreateView(outSqueeze, outSqueeze->GetViewShape(), outSqueeze->GetViewOffset());
    CHECK_RET(outView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outReFormat = l0op::ReFormat(outView, out->GetViewFormat());
    CHECK_RET(outReFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outCast = l0op::Cast(outReFormat, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(outCast, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIm2colBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIm2colBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
