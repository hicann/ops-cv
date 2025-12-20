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
 * \file grid_sampler2d.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_GRID_SAMPLER2D_H_
#define OP_API_INC_LEVEL0_GRID_SAMPLER2D_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *GridSampler2D(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, aclOpExecutor *executor);
}

#endif  // OP_API_INC_LEVEL0_GRID_SAMPLER2D_H_