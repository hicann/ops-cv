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
 * \file roi_align_grad.cpp
 * \brief RoiAlignGrad kernel entry.
 */

#include "roi_align_grad.h"

__global__ __aicore__ void roi_align_grad(GM_ADDR yDiff, GM_ADDR rois, GM_ADDR xDiff, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RoiAlignGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(RoiAlignGradTilingData, tilingData, tiling);

    NsRoiAlignGrad::RoiAlignGrad<float> op;
    op.Init(yDiff, rois, xDiff, workspace, &tilingData);
    op.Process();
}
