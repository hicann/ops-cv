/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gaussian_blur.h"
#include <array>
#include <cmath>
#include "../op_host/gaussian_blur_utils.h"
#include "gaussian_blur.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr size_t INPUT_RANK_MIN = 2;
static constexpr size_t INPUT_RANK_MAX = 3;
static constexpr size_t EXPECT_ATTR_SIZE = 2;

struct CanonicalAttrs {
    std::array<int64_t, EXPECT_ATTR_SIZE> ksize = {3, 3};
    float sigmaX = 0.0f;
    float sigmaY = 0.0f;
    int64_t borderType = gaussian_blur::BORDER_REFLECT_101;
};

static bool IsSupportedKernel(int64_t kernel)
{
    return kernel == 1 || kernel == 3 || kernel == 5 || kernel == 7 || kernel == 9 || kernel == 11 || kernel == 15 ||
           kernel == 21 || kernel == 31;
}

static bool CheckNotNull(const aclTensor* src, const aclIntArray* ksize, const aclTensor* dst)
{
    OP_CHECK_NULL(src, return false);
    OP_CHECK_NULL(ksize, return false);
    OP_CHECK_NULL(dst, return false);
    return true;
}

static bool CheckPlatformValid()
{
    const SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(socVersion == SocVersion::ASCEND950,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur only supports Ascend950."), return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* src, const aclTensor* dst)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(src, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(dst, src->GetDataType(), return false);
    return true;
}

static bool CheckFormatValid(const aclTensor* src, const aclTensor* dst)
{
    auto srcFormat = src->GetStorageFormat();
    auto dstFormat = dst->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(srcFormat) || IsPrivateFormat(dstFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format is not supported."), return false);
    OP_CHECK(srcFormat == dstFormat, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "src and dst format must be the same."),
             return false);
    OP_CHECK(srcFormat == op::Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur only supports ND public format."), return false);
    return true;
}

static bool HasSameShape(const aclTensor* src, const aclTensor* dst)
{
    const auto srcShape = src->GetViewShape();
    const auto dstShape = dst->GetViewShape();
    if (srcShape.GetDimNum() != dstShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < srcShape.GetDimNum(); ++i) {
        if (srcShape.GetDim(i) != dstShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool HasPositiveDims(const aclTensor* tensor)
{
    const auto shape = tensor->GetViewShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* src, const aclTensor* dst)
{
    const auto srcShape = src->GetViewShape();
    OP_CHECK(srcShape.GetDimNum() >= INPUT_RANK_MIN && srcShape.GetDimNum() <= INPUT_RANK_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "src only supports rank 2/3 ND image tensors."), return false);
    OP_CHECK(HasSameShape(src, dst), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "src and dst must have the same shape."),
             return false);
    OP_CHECK(HasPositiveDims(src) && HasPositiveDims(dst),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur does not support empty tensors."), return false);
    return true;
}

static bool CheckInplaceUnsupported(const aclTensor* src, const aclTensor* dst)
{
    OP_CHECK(src != dst, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur does not support in-place src and dst."),
             return false);
    const void* srcData = src->GetData();
    const void* dstData = dst->GetData();
    OP_CHECK(srcData == nullptr || dstData == nullptr || srcData != dstData,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur does not support aliased storage."), return false);
    return true;
}

static bool CheckPublicKernelSizeValid(const aclIntArray* ksize)
{
    const size_t size = ksize->Size();
    OP_CHECK(size == EXPECT_ATTR_SIZE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ksize must contain 2 elements."),
             return false);
    for (size_t i = 0; i < size; ++i) {
        const int64_t value = (*ksize)[i];
        OP_CHECK(gaussian_blur::IsExplicitKernelSizeValid(value),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ksize must be non-negative and odd when explicitly set."),
                 return false);
    }
    return true;
}

static bool CheckSigmaValid(double sigmaX, double sigmaY)
{
    OP_CHECK(std::isfinite(sigmaX) && std::isfinite(sigmaY), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sigma must be finite."),
             return false);
    OP_CHECK(sigmaX >= 0.0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sigmaX must be greater than or equal to 0."),
             return false);
    return true;
}

static bool CheckBorderTypeValid(int64_t borderType)
{
    int64_t canonicalBorderType = gaussian_blur::BORDER_REPLICATE;
    OP_CHECK(gaussian_blur::CanonicalizeBorderType(borderType, canonicalBorderType),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur does not support borderType=%ld.", borderType),
             return false);
    return true;
}

static void GetImageSize(const aclTensor* src, uint32_t& width, uint32_t& height)
{
    const auto shape = src->GetViewShape();
    height = static_cast<uint32_t>(shape.GetDim(0));
    width = static_cast<uint32_t>(shape.GetDim(1));
}

static aclnnStatus CanonicalizeAttrs(const aclTensor* src, const aclIntArray* ksize, double sigmaX, double sigmaY,
                                     int64_t borderType, CanonicalAttrs& canonical)
{
    uint32_t width = 0;
    uint32_t height = 0;
    GetImageSize(src, width, height);
    gaussian_blur::CanonicalParams params;
    OP_CHECK(
        gaussian_blur::CanonicalizeParams((*ksize)[0], (*ksize)[1], sigmaX, sigmaY, borderType, width, height, params),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur cannot canonicalize attributes."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(IsSupportedKernel(params.kernelW) && IsSupportedKernel(params.kernelH),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GaussianBlur supports K1/K3/K5/K7/K9/K11/K15/K21/K31, got [%ld,%ld].",
                     params.kernelW, params.kernelH),
             return ACLNN_ERR_PARAM_INVALID);

    canonical.ksize[0] = params.kernelW;
    canonical.ksize[1] = params.kernelH;
    canonical.sigmaX = static_cast<float>(params.sigmaX);
    canonical.sigmaY = static_cast<float>(params.sigmaY);
    canonical.borderType = params.borderType;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* src, const aclIntArray* ksize, double sigmaX, double sigmaY,
                               int64_t borderType, const aclTensor* dst, CanonicalAttrs& canonical)
{
    CHECK_RET(CheckNotNull(src, ksize, dst), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckPlatformValid(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(src, dst), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(src, dst), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(src, dst), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInplaceUnsupported(src, dst), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPublicKernelSizeValid(ksize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSigmaValid(sigmaX, sigmaY), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBorderTypeValid(borderType), ACLNN_ERR_PARAM_INVALID);
    return CanonicalizeAttrs(src, ksize, sigmaX, sigmaY, borderType, canonical);
}

} // namespace

extern "C" {

aclnnStatus aclnnGaussianBlurGetWorkspaceSize(const aclTensor* src, const aclIntArray* ksize, double sigmaX,
                                              double sigmaY, int64_t borderType, const aclTensor* dst,
                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGaussianBlur, DFX_IN(src, ksize, sigmaX, sigmaY, borderType), DFX_OUT(dst));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CanonicalAttrs canonical;
    auto ret = CheckParams(src, ksize, sigmaX, sigmaY, borderType, dst, canonical);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto* canonicalKsize = uniqueExecutor->AllocIntArray(canonical.ksize.data(), canonical.ksize.size());
    CHECK_RET(canonicalKsize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto* kernelSrc = l0op::Contiguous(src, uniqueExecutor.get());
    CHECK_RET(kernelSrc != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::GaussianBlur(kernelSrc, canonicalKsize, canonical.sigmaX, canonical.sigmaY,
                                     canonical.borderType, const_cast<aclTensor*>(dst), uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGaussianBlur(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclTensor* src,
                              const aclIntArray* ksize, double sigmaX, double sigmaY, int64_t borderType,
                              aclTensor* dst, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGaussianBlur);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    CanonicalAttrs canonical;
    auto ret = CheckParams(src, ksize, sigmaX, sigmaY, borderType, dst, canonical);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
