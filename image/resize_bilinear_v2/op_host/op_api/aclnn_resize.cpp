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
 * \file aclnn_resize.cpp
 * \brief aclnn_resize
 */

#include "aclnn_resize.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transdata.h"
#include "image/resize_nearest_neighbor_v2/op_host/op_api/resize_nearest_neighbor_v2.h"
#include "resize_bilinear_v2.h"
#include "acl/acl.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_CHANNEL = 1;
static constexpr size_t DIM_HEIGHT = 2;
static constexpr size_t DIM_WIDTH = 3;

static constexpr size_t DIM_BATCH_NCHW = 0;
static constexpr size_t DIM_CHANNEL_NCHW = 1;
static constexpr size_t DIM_HEIGHT_NCHW = 2;
static constexpr size_t DIM_WIDTH_NCHW = 3;
static constexpr size_t DIM_BATCH_NHWC = 0;
static constexpr size_t DIM_HEIGHT_NHWC = 1;
static constexpr size_t DIM_WIDTH_NHWC = 2;
static constexpr size_t DIM_CHANNEL_NHWC = 3;
static constexpr size_t SCALES_SIZE = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DATA = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DATA_ASCEND910_95 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckShape(const aclTensor *self, const aclFloatArray *scales, const aclTensor *out, bool extendFlag)
{
    // The scale parameter is retained for interface consistency and is currently not used in computations.
    auto selfShape = self->GetViewShape();
    auto outShape = out->GetViewShape();
    if (selfShape.GetDimNum() != NCHW_DIM_NUM || outShape.GetDimNum() != NCHW_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The selfShape and outShape must be 4 dim.");
        return false;
    }
    uint64_t size = 0;
    auto ret = aclGetFloatArraySize(scales, &size);
    if (ret != 0 || size != NCHW_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclGetFloatArraySize error, ret=%lu, size=%lu, NCHW_DIM_NUM=%zu.",
            static_cast<uint64_t>(ret),
            size,
            NCHW_DIM_NUM);
        return false;
    }

    size_t dimBatch = DIM_BATCH_NCHW;
    size_t dimChanel = DIM_CHANNEL_NCHW;
    size_t dimHeight = DIM_HEIGHT_NCHW;
    size_t dimWidth = DIM_WIDTH_NCHW;
    if (extendFlag && self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        dimBatch = DIM_BATCH_NHWC;
        dimChanel = DIM_CHANNEL_NHWC;
        dimHeight = DIM_HEIGHT_NHWC;
        dimWidth = DIM_WIDTH_NHWC;
    }

    if (selfShape.GetDim(dimBatch) != outShape.GetDim(dimBatch) ||
        selfShape.GetDim(dimChanel) != outShape.GetDim(dimChanel)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of batch or channel is not matched.");
        return false;
    }

    if (extendFlag) {
        int64_t dstDimHeight = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(dimHeight)) * static_cast<double>((*scales)[dimHeight]));
        if (outShape.GetDim(dimHeight) != dstDimHeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of height is not matched.");
            return false;
        }

        int64_t dstDimWidth = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(dimWidth)) * static_cast<double>((*scales)[dimWidth]));
        if (outShape.GetDim(dimWidth) != dstDimWidth) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of width is not matched.");
            return false;
        }
    }

    return true;
}

static bool CheckNotNull(const aclTensor *self, const aclFloatArray *scales, const char *mode, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(out, return false);
    if (mode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode is null, please check input arguments");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out, bool extendFlag)
{
    if (extendFlag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_DATA_ASCEND910_95, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_DATA, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckModeStr(const char *mode)
{
    if (strncmp(mode, "nearest", strlen("nearest")) == 0 || strncmp(mode, "bilinear", strlen("bilinear")) == 0) {
        return true;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckModeStr failed, mode:%s", mode);
    return false;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out, bool extendFlag)
{
    if (extendFlag) {
        auto xFormat = self->GetStorageFormat();
        auto yFormat = out->GetStorageFormat();
        if (xFormat != yFormat) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The format of self and out must be same.");
            return false;
        }

        if (xFormat != op::Format::FORMAT_NCHW && xFormat != op::Format::FORMAT_NHWC) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The format of self must be NCHW or NHWC.");
            return false;
        }

        return true;
    }

    if (self->GetStorageFormat() != op::Format::FORMAT_NCHW || out->GetStorageFormat() != op::Format::FORMAT_NCHW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The format of self or out must be NCHW.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *self, const aclFloatArray *scales, const char *mode, const aclTensor *out, bool extendFlag)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, scales, mode, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查mode是否支持
    CHECK_RET(CheckModeStr(mode), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValid(self, out, extendFlag), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入格式是否为NCHW
    CHECK_RET(CheckFormat(self, out, extendFlag), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查输入tensor的shape
    CHECK_RET(CheckShape(self, scales, out, extendFlag), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor *CreateSizesV35(const aclTensor *out, aclOpExecutor *executor)
{
    auto outShape = op::ToShapeVector(out->GetViewShape());
    const aclIntArray *arr = executor->AllocIntArray(outShape.data(), outShape.size());
    auto sizes = executor->ConvertToTensor(arr, op::ToOpDataType(ACL_INT32));
    return sizes;
}

static const aclTensor *CreateSizesAscend910_95(
    const aclTensor *self, const aclFloatArray *scales, aclOpExecutor *executor)
{
    uint64_t size = 0;
    auto ret = aclGetFloatArraySize(scales, &size);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales is null, please check input arguments of scales");
        return nullptr;
    }
    if (self->GetViewShape().GetDimNum() != size) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "dim num of self should equal with size of scales, self DimNum=%zu, size=%lu",
            self->GetViewShape().GetDimNum(),
            size);
        return nullptr;
    }
    auto selfShape = self->GetViewShape();
    std::vector<int64_t> sizesList(SCALES_SIZE);
    auto xFormat = self->GetStorageFormat();
    if (xFormat == op::Format::FORMAT_NCHW) {
        sizesList[0] = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(DIM_HEIGHT_NCHW)) * static_cast<double>((*scales)[DIM_HEIGHT_NCHW]));
        sizesList[1] = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(DIM_WIDTH_NCHW)) * static_cast<double>((*scales)[DIM_WIDTH_NCHW]));
    } else {
        sizesList[0] = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(DIM_HEIGHT_NHWC)) * static_cast<double>((*scales)[DIM_HEIGHT_NHWC]));
        sizesList[1] = static_cast<int64_t>(
            static_cast<double>(selfShape.GetDim(DIM_WIDTH_NHWC)) * static_cast<double>((*scales)[DIM_WIDTH_NHWC]));
    }
    const aclIntArray *arr = executor->AllocIntArray(sizesList.data(), sizesList.size());
    auto sizes = executor->ConvertToTensor(arr, op::ToOpDataType(ACL_INT32));
    return sizes;
}

static bool GetExtendPathFlag()
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_95) {
        return true;
    }
    return false;
}

aclnnStatus aclnnResizeGetWorkspaceSize(const aclTensor *self, const aclFloatArray *scales, const char *mode,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnResize, DFX_IN(self, scales, mode), DFX_OUT(out));
    // 参数检查
    bool extendFlag = GetExtendPathFlag();

    auto ret = CheckParams(self, scales, mode, out, extendFlag);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (extendFlag) {
        auto sizes = CreateSizesAscend910_95(self, scales, uniqueExecutor.get());
        CHECK_RET(sizes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 不必转到5HD，直接执行L0算子 带scales参数
        const aclTensor *resizeRet = nullptr;
        if (strncmp(mode, "nearest", strlen("nearest")) == 0) {
            resizeRet = l0op::ResizeNearestNeighborV2(self, sizes, nullptr, outContiguous, uniqueExecutor.get());
        } else {
            resizeRet = l0op::ResizeBilinearV2With4d(self, sizes, false, nullptr, outContiguous, uniqueExecutor.get());
        }
        CHECK_RET(resizeRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将计算结果拷贝到输出输出上
        auto viewCopyResult = l0op::ViewCopy(resizeRet, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto sizes = CreateSizesV35(out, uniqueExecutor.get());
        CHECK_RET(sizes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfData = l0op::TransDataSpecial(self, Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
        CHECK_RET(selfData != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto outData = l0op::TransDataSpecial(outContiguous, Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
        CHECK_RET(outData != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 执行L0算子
        const aclTensor *resizeRet = nullptr;
        if (strncmp(mode, "nearest", strlen("nearest")) == 0) {
            resizeRet = l0op::ResizeNearestNeighborV2(selfData, sizes, nullptr, outData, uniqueExecutor.get());
        } else if (strncmp(mode, "bilinear", strlen("bilinear")) == 0) {
            resizeRet = l0op::ResizeBilinearV2(selfData, sizes, false, outData, uniqueExecutor.get());
        }

        CHECK_RET(resizeRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 输出转NCHW
        auto outRet = l0op::TransData(resizeRet, self->GetStorageFormat(), 0, uniqueExecutor.get());
        CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将计算结果拷贝到输出输出上
        auto viewCopyResult = l0op::ViewCopy(outRet, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnResize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnResize);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
