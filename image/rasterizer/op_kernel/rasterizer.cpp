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
 * \file rasterizer.cpp
 * \brief
 */

#include "rasterizer.h"
#include "barycentric_from_imgcoord_kernel.h"

extern "C" __global__ __aicore__ void rasterizer(GM_ADDR v, GM_ADDR f, GM_ADDR d, GM_ADDR findices, GM_ADDR barycentric,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        {
            NsRasterizer::Rasterizer<float> op;
            op.Init(v, f, d, findices, barycentric, userWS, &tilingData);
            op.Process();
        }
        AscendC::SyncAll();
        BarycentricFromImgcoord::BarycentricFromImgcoordAIV<float> op;
        op.Init(v, f, findices, barycentric, userWS, &tilingData);
        op.Process();
    }
}