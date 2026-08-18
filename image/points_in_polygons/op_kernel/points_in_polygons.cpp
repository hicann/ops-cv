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
 * \file points_in_polygons.cpp
 * \brief PointsInPolygons AscendC kernel entry
 */

#include "kernel_operator.h"
#include "arch35/points_in_polygons_tiling_key.h"
#include "arch35/points_in_polygons_tiling_data.h"
#include "arch35/points_in_polygons.h"

template <int KEY>
__global__ __aicore__ void points_in_polygons(GM_ADDR points, GM_ADDR polygons, GM_ADDR output, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PointsInPolygonsTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA(td, tiling);

    PointsInPolygonsKernel<DTYPE_POINTS, KEY> op;
    op.Init(points, polygons, output, &td);
    op.Process();
}
