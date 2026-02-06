/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2_grad.h
 * \brief resize_nearest_neighbor_v2_grad
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_NEAREST_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_NEAREST_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *ResizeNearestNeighborV2Grad5Hd(const aclTensor *grads, const aclIntArray *inputSize, bool alignCorners,
    bool halfPixelCenters, aclOpExecutor *executor);

const aclTensor *ResizeNearestNeighborV2Grad(const aclTensor *grads, const aclIntArray *inputSize, bool alignCorners,
    bool halfPixelCenters, const aclFloatArray *scales, aclOpExecutor *executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_NEAREST_GRAD_OP_H_