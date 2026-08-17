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
 * \file dilation2_d_backprop_filter_apt.cpp
 * \brief Kernel entry for dilation2_d_backprop_filter operator
 *
 * Template parameter:
 *   schMode (uint32_t): scene mode (NORMAL=0 only)
 *   dtype is handled by DTYPE_X macro (auto-instantiated per dtype).
 *
 * Kernel entry params (order matches REG_OP INPUTs + OUTPUTs):
 *   x, filter, out_backprop, y, workspace, tiling
 *
 * v2.1: user workspace restored for per-core deterministic accumulation buffer.
 *       workspace provides needCoreNum × perCoreBufElems × sizeof(float) buffer.
 */

#include "arch35/dilation2_d_backprop_filter_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void dilation2_d_backprop_filter(GM_ADDR x, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Dilation2DBackpropFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(Dilation2DBackpropFilterTilingData, tilingData, tiling);

    // DTYPE_ macro auto-instantiates per dtype combination
    // v2.0: only float32, DTYPE_X expands to float
    __gm__ DTYPE_X* xGm = (__gm__ DTYPE_X*)x;
    __gm__ DTYPE_X* filterGm = (__gm__ DTYPE_X*)filter;
    __gm__ DTYPE_X* outBackpropGm = (__gm__ DTYPE_X*)out_backprop;
    __gm__ DTYPE_X* yGm = (__gm__ DTYPE_X*)y;

    // v2.1: get user workspace for per-core deterministic accumulation buffer
    __gm__ DTYPE_X* wsPerCore = (__gm__ DTYPE_X*)AscendC::GetUserWorkspace(workspace);
    if (wsPerCore == nullptr) {
        return;
    }

    if constexpr (schMode == DILATION2D_BACKPROP_FILTER_MODE_NORMAL) {
        NsDilation2DBackpropFilter::Process<DTYPE_X>(&tilingData, xGm, filterGm, outBackpropGm, yGm, wsPerCore);
    }
}
