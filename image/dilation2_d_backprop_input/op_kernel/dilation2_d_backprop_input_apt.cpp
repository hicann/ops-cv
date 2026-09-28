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
 * \file dilation2_d_backprop_input_apt.cpp
 * \brief Kernel entry for dilation2_d_backprop_input operator
 */

#include "arch35/dilation2_d_backprop_input_simt.h"
#include "arch35/dilation2_d_backprop_input_simt_nondet.h"

template <uint32_t schMode, uint32_t deterministic>
__global__ __aicore__ void dilation2_d_backprop_input(GM_ADDR x, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Dilation2DBackpropInputTilingData);
    GET_TILING_DATA_WITH_STRUCT(Dilation2DBackpropInputTilingData, tilingData, tiling);

    if constexpr (deterministic == DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_NO) {
        constexpr int64_t kIdxInt32Max = 0x7fffffff;
        const bool fitsInt32 = tilingData.outTotalNum <= kIdxInt32Max && tilingData.inTotalNum <= kIdxInt32Max &&
                               (tilingData.outputH * tilingData.strideH + tilingData.filterH * tilingData.rateH) <=
                                   kIdxInt32Max &&
                               (tilingData.outputW * tilingData.strideW + tilingData.filterW * tilingData.rateW) <=
                                   kIdxInt32Max &&
                               tilingData.filterH * tilingData.filterW * tilingData.depth <= kIdxInt32Max;
        if (fitsInt32) {
            NsDilation2DBackpropInputNonDet::Process<DTYPE_X, schMode, int32_t>(x, filter, out_backprop, y, workspace,
                                                                                tiling, &tilingData);
        } else {
            NsDilation2DBackpropInputNonDet::Process<DTYPE_X, schMode, int64_t>(x, filter, out_backprop, y, workspace,
                                                                                tiling, &tilingData);
        }
    } else {
        NsDilation2DBackpropInput::Process<DTYPE_X, schMode>(x, filter, out_backprop, y, workspace, tiling,
                                                             &tilingData);
    }
}
