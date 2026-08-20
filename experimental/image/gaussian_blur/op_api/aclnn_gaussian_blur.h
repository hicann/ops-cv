/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GAUSSIAN_BLUR_H_
#define OP_API_INC_GAUSSIAN_BLUR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates the workspace size required by aclnnGaussianBlur.
 * @param [in] src Input FLOAT32 ND image with shape [H, W] or [H, W, C].
 * @param [in] ksize Gaussian kernel size [kernelWidth, kernelHeight].
 * @param [in] sigmaX Horizontal Gaussian standard deviation.
 * @param [in] sigmaY Vertical Gaussian standard deviation.
 * @param [in] borderType OpenCV-compatible border mode.
 * @param [in] dst Output tensor with the same metadata as src.
 * @param [out] workspaceSize Required workspace size in bytes.
 * @param [out] executor Operator executor.
 * @return aclnnStatus Execution status.
 */
ACLNN_API aclnnStatus aclnnGaussianBlurGetWorkspaceSize(const aclTensor* src, const aclIntArray* ksize, double sigmaX,
                                                        double sigmaY, int64_t borderType, const aclTensor* dst,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Executes GaussianBlur on the specified stream.
 * @param [in] workspace Device workspace address.
 * @param [in] workspaceSize Workspace size returned by aclnnGaussianBlurGetWorkspaceSize.
 * @param [in] executor Operator executor returned by aclnnGaussianBlurGetWorkspaceSize.
 * @param [in] src Input tensor.
 * @param [in] ksize Gaussian kernel size.
 * @param [in] sigmaX Horizontal Gaussian standard deviation.
 * @param [in] sigmaY Vertical Gaussian standard deviation.
 * @param [in] borderType OpenCV-compatible border mode.
 * @param [out] dst Output tensor.
 * @param [in] stream ACL runtime stream.
 * @return aclnnStatus Execution status.
 */
ACLNN_API aclnnStatus aclnnGaussianBlur(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclTensor* src, const aclIntArray* ksize, double sigmaX, double sigmaY,
                                        int64_t borderType, aclTensor* dst, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GAUSSIAN_BLUR_H_
