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
 * Template parameters:
 *   detMode (uint32_t): 0=DETERMINISTIC, 1=NON_DETERMINISTIC (compile-time dispatch)
 *   schMode (uint32_t): 0=NORMAL (reserved)
 *   dtype is handled by DTYPE_X macro (auto-instantiated per dtype).
 *
 * detMode=0: NsDilation2DBackpropFilter::Process (dilation2_d_backprop_filter_simt.h)
 *   - 4-phase: ZeroOut wsBuf → Compute += perThreadBuf → TreeReduce → ReduceCores → yGm
 * detMode=1: NsDilation2DBackpropFilterNonDet::Process (dilation2_d_backprop_filter_simt_nondet.h)
 *   - 2-phase: ZeroOut yGm → SyncAll → atomic_add to yGm
 *   - IDX_T dual-path: uint32_t (totalElements <= UINT32_MAX) / uint64_t (overflow safe)
 */

#include "arch35/dilation2_d_backprop_filter_simt.h"
#include "arch35/dilation2_d_backprop_filter_simt_nondet.h"

template <uint32_t detMode, uint32_t schMode>
__global__ __aicore__ void dilation2_d_backprop_filter(GM_ADDR x, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    __gm__ DTYPE_X* xGm = (__gm__ DTYPE_X*)x;
    __gm__ DTYPE_X* filterGm = (__gm__ DTYPE_X*)filter;
    __gm__ DTYPE_X* outBackpropGm = (__gm__ DTYPE_X*)out_backprop;
    __gm__ DTYPE_X* yGm = (__gm__ DTYPE_X*)y;

    if constexpr (detMode == DILATION2D_BACKPROP_FILTER_MODE_DETERMINISTIC) {
        REGISTER_TILING_DEFAULT(Dilation2DBackpropFilterTilingData);
        GET_TILING_DATA_WITH_STRUCT(Dilation2DBackpropFilterTilingData, tilingData, tiling);
        __gm__ DTYPE_X* wsPerCore = (__gm__ DTYPE_X*)AscendC::GetUserWorkspace(workspace);
        if (wsPerCore == nullptr) {
            return;
        }
        NsDilation2DBackpropFilter::Process<DTYPE_X>(&tilingData, xGm, filterGm, outBackpropGm, yGm, wsPerCore);
    } else {
        REGISTER_TILING_DEFAULT(Dilation2DBackpropFilterTilingData);
        GET_TILING_DATA_WITH_STRUCT(Dilation2DBackpropFilterTilingData, tilingData, tiling);
        if (tilingData.totalElements <= static_cast<int64_t>(UINT32_MAX)) {
            if (tilingData.isNCHW == 1) {
                NsDilation2DBackpropFilterNonDet::Process<DTYPE_X, true, uint32_t>(&tilingData, xGm, filterGm,
                                                                                   outBackpropGm, yGm);
            } else {
                NsDilation2DBackpropFilterNonDet::Process<DTYPE_X, false, uint32_t>(&tilingData, xGm, filterGm,
                                                                                    outBackpropGm, yGm);
            }
        } else {
            if (tilingData.isNCHW == 1) {
                NsDilation2DBackpropFilterNonDet::Process<DTYPE_X, true, uint64_t>(&tilingData, xGm, filterGm,
                                                                                   outBackpropGm, yGm);
            } else {
                NsDilation2DBackpropFilterNonDet::Process<DTYPE_X, false, uint64_t>(&tilingData, xGm, filterGm,
                                                                                    outBackpropGm, yGm);
            }
        }
    }
}
