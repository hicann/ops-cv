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
 * \file roi_pooling_grad_with_arg_max_apt.cpp
 * \brief
 */

#include "arch35/roi_pooling_grad_with_arg_max_simt.h"
#include "arch35/roi_pooling_grad_with_arg_max_tiling_data.h"
#include "arch35/roi_pooling_grad_with_arg_max_tiling_key.h"
using namespace AscendC;
using namespace RoiPoolingGradWithArgMaxOps;

template <uint64_t dType>
__global__ __aicore__ void roi_pooling_grad_with_arg_max(GM_ADDR grad, GM_ADDR x, GM_ADDR rois, GM_ADDR roi_actual_num, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(RoiPoolingGradWithArgMaxRegBaseTilingData);
    GET_TILING_DATA(tilingDataIn, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe tpipe;

    if constexpr (dType == ROI_POOLING_GRAD_WITH_ARG_MAX_TPL_FP32) {
        const RoiPoolingGradWithArgMaxRegBaseTilingData* __restrict tilingData = &tilingDataIn;
        RoiPoolingGradWithArgMaxSimt<float, float> roiPoolingGradWithArgMaxKernel;
        roiPoolingGradWithArgMaxKernel.Init(grad, x, rois, argmax, y, userWS, tilingData);
        roiPoolingGradWithArgMaxKernel.Process();
    } 
    else if constexpr (dType == ROI_POOLING_GRAD_WITH_ARG_MAX_TPL_FP16) {
        const RoiPoolingGradWithArgMaxRegBaseTilingData* __restrict tilingData = &tilingDataIn;
        RoiPoolingGradWithArgMaxSimt<float, half> roiPoolingGradWithArgMaxKernel;
        roiPoolingGradWithArgMaxKernel.Init(grad, x, rois, argmax, y, userWS, tilingData);
        roiPoolingGradWithArgMaxKernel.Process();
    }
}


