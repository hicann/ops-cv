/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest_exact2d_grad_apt.cpp
 * \brief upsample_nearest_exact2d_grad_apt
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "./arch35/upsample_nearest_exact2d_grad_data_copy.h"
#include "./arch35/upsample_nearest_exact2d_grad_simt.h"
#include "./arch35/upsample_nearest_exact2d_grad_tiling_key.h"
#include "./arch35/upsample_nearest_exact2d_grad_tiling_data.h"

using namespace AscendC;
using namespace UpsampleNearestExact2dGrad;

template <uint64_t schId, uint64_t isUint32>
__global__ __aicore__ void upsample_nearest_exact2d_grad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(UpsampleNearestExact2dGradRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleNearestExact2dGradRegBaseTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    TPipe pipe;
    if constexpr (schId == 0) {
        NearestExact2dGradDataCopy<DTYPE_GRAD_OUTPUT> op;
        op.Init(x, y, &pipe, &tilingData);
        op.Process();
        return;
    } else {
        // upsample_nearest_exact2d_grad 算子固定使用 exact 模式
        constexpr bool isExact = true;
        if constexpr (isUint32 == 1) {
            NearestExact2dGradSimt<DTYPE_GRAD_OUTPUT, uint32_t, isExact, schId> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else {
            NearestExact2dGradSimt<DTYPE_GRAD_OUTPUT, uint64_t, isExact, schId> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        return;
    }
}
