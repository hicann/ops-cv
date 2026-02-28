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
 * \file upsample_bilinear2d_aa_backward_apt.cpp
 * \brief
 */

#include "./arch35/upsample_bilinear2d_aa_backward_data_copy.h"
#include "./arch35/upsample_bilinear2d_aa_backward_simt.h"
#include "./arch35/upsample_bilinear2d_aa_backward_tiling_key.h"
#include "./arch35/upsample_bilinear2d_aa_backward_tiling_data.h"

using namespace UpsampleBilinear2dAABackward;

template <uint64_t schId, uint64_t isInt32, uint64_t isDetermine>
__global__ __aicore__ void upsample_bilinear2d_aa_backward(
    GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(UpsampleBilinear2dAABackwardRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleBilinear2dAABackwardRegBaseTilingData, tilingData, tiling);

    TPipe pipe;
    if constexpr (schId == 0) {
        Bilinear2dAABackwardDataCopy<DTYPE_GRAD_OUTPUT> op;
        op.Init(grad_output, grad_input, &pipe, &tilingData);
        op.Process();
    } else {
        if constexpr(isInt32 == 1) {
            Bilinear2dAABackwardSimt<DTYPE_GRAD_OUTPUT, uint32_t, int32_t, isDetermine> op;
            op.Init(grad_output, grad_input, &tilingData);
            op.Process();
        } else {
            Bilinear2dAABackwardSimt<DTYPE_GRAD_OUTPUT, uint64_t, int64_t, isDetermine> op;
            op.Init(grad_output, grad_input, &tilingData);
            op.Process();
        }
    }
}
