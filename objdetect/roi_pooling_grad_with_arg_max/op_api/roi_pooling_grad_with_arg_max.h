/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_grad_with_arg_max.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_ROI_POOLING_GRAD_WITH_ARG_MAX_H_
#define OP_API_INC_LEVEL0_OP_ROI_POOLING_GRAD_WITH_ARG_MAX_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *RoiPoolingGradWithArgMax(const aclTensor *grad,
    const aclTensor *x,
    const aclTensor *rois,
    const aclTensor *roiActualNumOptional,
    const aclTensor *argmax,
    int64_t pooledH,
    int64_t pooledW,
    double spatialScaleH,
    double spatialScaleW,
    int64_t poolChannel,
    aclOpExecutor *executor);
}

#endif
