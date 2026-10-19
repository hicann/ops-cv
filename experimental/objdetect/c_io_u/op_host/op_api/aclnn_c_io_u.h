/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_C_IO_U_H_
#define OP_API_INC_LEVEL2_ACLNN_C_IO_U_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Experimental CIoU first-stage API (workspace size).
 * @domain aclnn_ops_infer
 *
 * Extends the business aclnnCIoU contract with an explicit atanSubFlag attribute so the
 * experimental CIoU kernel can skip the atan branch when requested.
 *
 * @param [in] bBoxes: device aclTensor, predicted boxes, FLOAT/FLOAT16, ND, shape [4, N].
 * @param [in] gtBoxes: device aclTensor, ground-truth boxes, FLOAT/FLOAT16, ND, shape [4, N].
 * @param [in] trans: true for xywh, false for xyxy.
 * @param [in] isCross: cross-pair mode; currently only false is supported.
 * @param [in] mode: "iou" or "iof".
 * @param [in] atanSubFlag: when false, atan_sub output is filled with zeros.
 * @param [out] overlap: device aclTensor, FLOAT/FLOAT16, ND, shape [1, N] when isCross is false.
 * @param [out] atanSub: device aclTensor, FLOAT/FLOAT16, ND, shape [1, N] when isCross is false.
 * @param [out] workspaceSize: required workspace size in bytes.
 * @param [out] executor: op executor.
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnCIoUGetWorkspaceSize(const aclTensor* bBoxes, const aclTensor* gtBoxes, bool trans,
                                                bool isCross, const char* mode, bool atanSubFlag, aclTensor* overlap,
                                                aclTensor* atanSub, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Experimental CIoU second-stage API (execute).
 */
ACLNN_API aclnnStatus aclnnCIoU(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_C_IO_U_H_
