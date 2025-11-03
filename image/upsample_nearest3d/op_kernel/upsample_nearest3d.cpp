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
 * \file upsample_nearest3d.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200
#include "upsample_nearest3d_310p.h"
#else
#include "upsample_nearest3d.h"
#endif
#include "upsample_nearest3d_struct.h"

using namespace UpsampleNearest3d;

template <int D_T_X, int D_T_Y>
__global__ __aicore__ void upsample_nearest3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if __CCE_AICORE__ == 200
    GM_ADDR userWS = GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(UpsampleNearest3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleNearest3dTilingData, tilingData, tiling);
    UpsampleNearest3d310pKernelImpl<D_T_X, D_T_Y>(x, y, false, userWS, &tilingData);
#else
    REGISTER_TILING_DEFAULT(UpsampleNearest3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleNearest3dTilingData, tilingData, tiling);
    UpsampleNearest3dKernelImpl<D_T_X, D_T_Y>(x, y, false, &tilingData);
#endif
}
