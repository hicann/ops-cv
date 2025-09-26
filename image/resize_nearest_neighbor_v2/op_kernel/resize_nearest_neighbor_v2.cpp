/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2.cpp
 * \brief resize_nearest_neighbor_v2
 */
#include "./arch35/resize_nearest_neighbor_v2_base.h"
#include "./arch35/resize_nearest_neighbor_v2_simt.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_big_c.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_jh.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_small_c.h"
#include "./arch35/resize_nearest_neighbor_v2_tiling_key.h"

using namespace AscendC;

template <uint64_t schId, uint64_t format, uint64_t alignCorners, uint64_t halfPixelCenters, uint64_t idxInt32>
__global__ __aicore__ void resize_nearest_neighbor_v2(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_SMALL_C) {
        ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_S_C<DTYPE_X> op;
        op.Init(x, size, y, userWS, &tilingData);
        op.Process();
        return;
    }

    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_BIG_C) {
        ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_BIG_C<DTYPE_X> op;
        op.Init(x, size, y, userWS, &tilingData);
        op.Process();
        return;
    }

    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_AGGR_C) {
        ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_JH<DTYPE_X> op;
        op.Init(x, size, y, userWS, &tilingData);
        op.Process();
        return;
    }

    if constexpr (idxInt32) {
        ResizeNearestNeighborV2::
            ResizeNearestNeighborV2Simt<DTYPE_X, uint32_t, format, schId, alignCorners, halfPixelCenters>
                op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    } else {
        ResizeNearestNeighborV2::
            ResizeNearestNeighborV2Simt<DTYPE_X, uint64_t, format, schId, alignCorners, halfPixelCenters>
                op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    }
}
