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
 * \file deformable_roi_pool.cpp
 * \brief Kernel entry for deformable_roi_pool operator
 */

#include "arch35/deformable_roi_pool_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void deformable_roi_pool(GM_ADDR x, GM_ADDR rois, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DeformableRoiPoolTilingData);
    GET_TILING_DATA_WITH_STRUCT(DeformableRoiPoolTilingData, tilingData, tiling);

    const __gm__ DTYPE_X* xGm = (const __gm__ DTYPE_X*)x;
    const __gm__ DTYPE_X* roisGm = (const __gm__ DTYPE_X*)rois;
    __gm__ DTYPE_X* yGm = (__gm__ DTYPE_X*)y;

    if constexpr (schMode == DEFORMABLE_ROI_POOL_TPL_WITH_OFFSET) {
        const __gm__ DTYPE_X* offsetGm = (const __gm__ DTYPE_X*)offset;
        NsDeformableRoiPool::DeformableRoiPoolProcess<DTYPE_X, true>(&tilingData, xGm, roisGm, offsetGm, yGm);
    } else {
        // HAS_OFFSET=false leaves the dummy pointer unused.
        NsDeformableRoiPool::DeformableRoiPoolProcess<DTYPE_X, false>(&tilingData, xGm, roisGm, xGm, yGm);
    }
}
