/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_apt.cpp
 * \brief ResizeUpsampleTrilinear APT kernel entry function for A5
 */
#include "./arch35/resize_upsample_trilinear_tiling_key.h"
#include "./arch35/resize_upsample_trilinear_tiling_data.h"
#include "./arch35/resize_upsample_trilinear_simt.h"
#include "./arch35/resize_upsample_trilinear_simt_base.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

template <uint32_t dtypeKey>
__global__ __aicore__ void resize_upsample_trilinear(GM_ADDR output, GM_ADDR input, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ResizeUpsampleTrilinearArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ResizeUpsampleTrilinearArch35TilingData, tilingData, tiling);
    AscendC::InitSocState();

    if (TILING_KEY_IS(TPL_DTYPE_FP32)) {
        ResizeUpsampleTrilinear::ResizeUpsampleTrilinearSimt<float> op;
        op.Init(output, input, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TPL_DTYPE_FP16)) {
        ResizeUpsampleTrilinear::ResizeUpsampleTrilinearSimt<half> op;
        op.Init(output, input, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TPL_DTYPE_BF16)) {
        ResizeUpsampleTrilinear::ResizeUpsampleTrilinearSimt<bfloat16_t> op;
        op.Init(output, input, &tilingData);
        op.Process();
    }
}