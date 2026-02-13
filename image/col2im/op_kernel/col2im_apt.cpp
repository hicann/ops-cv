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
 * \file col2im_apt.cpp
 * \brief
 */

#include "arch35/col2im_simt.h"
#include "arch35/col2im_tiling_data.h"
#include "arch35/col2im_tiling_key.h"
using namespace AscendC;
using namespace Col2imOps;

template <uint64_t dType>
__global__ __aicore__ void col2im(GM_ADDR gradOut, GM_ADDR outputSize, GM_ADDR gradIn, GM_ADDR workspace, GM_ADDR tiling)
{
    // 仅AIV支持simt，此处限制之用V核
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(Col2imRegBaseTilingData);
    GET_TILING_DATA(tilingDataIn, tiling);

    if constexpr (dType == COL2IM_TPL_FP32) {
        const Col2imRegBaseTilingData* __restrict tilingData = &tilingDataIn;
        Col2imSimt<float, float> col2imKernel;
        col2imKernel.Init(gradIn, gradOut, workspace, tilingData);
        col2imKernel.Process();
    } else if constexpr (dType == COL2IM_TPL_FP16) {
        const Col2imRegBaseTilingData* __restrict tilingData = &tilingDataIn;
        Col2imSimt<float, half> col2imKernel;
        col2imKernel.Init(gradIn, gradOut, workspace, tilingData);
        col2imKernel.Process();
    } else if constexpr (dType == COL2IM_TPL_BF16) {
        const Col2imRegBaseTilingData* __restrict tilingData = &tilingDataIn;
        Col2imSimt<float, bfloat16_t> col2imKernel;
        col2imKernel.Init(gradIn, gradOut, workspace, tilingData);
        col2imKernel.Process();
    }
}


