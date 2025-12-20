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
 * \file upsample_linear1d.cpp
 * \brief
 */

#include "upsample_linear1d.h"

using namespace UpsampleLinear1d;

extern "C" __global__ __aicore__ void upsample_linear1d(
    GM_ADDR input, GM_ADDR size, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    const UpsampleLinear1dTilingData *__restrict tiling_data = &tilingData;
    const TCubeTiling *__restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
    const TCubeTiling *__restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);

    if (TILING_KEY_IS(1)) {
        if (tiling_data->dataType == 2) {
            UpsampleLinear1dND<float> op;
            REGIST_MATMUL_OBJ(
                &op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
            op.Init(input, output, userWS, &tilingData);
            op.Process();
        }
    }
}
