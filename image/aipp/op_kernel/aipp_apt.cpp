/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aipp_apt.cpp
 * \brief aipp kernel main
 */

#include "arch35/aipp_kernel.h"
#include "arch35/aipp_rgb_yuv.h"

#define FORMAT_RGB_INDICE_UINT32 1
#define FORMAT_RGB_2_YUV_INDICE_UINT32 3

using namespace Aipp_Kernel;

extern "C" __global__ __aicore__ void Aipp(
    GM_ADDR images,GM_ADDR params, GM_ADDR features, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(AippTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(TILING_KEY_IS(FORMAT_RGB_INDICE_UINT32)) {
        Aipp_Kernel::AippRgb<DTYPE_FEATURES, uint32_t> op;
        op.Init(images, features, tilingData);
        op.Process(tiling);     
    } else if (TILING_KEY_IS(FORMAT_RGB_2_YUV_INDICE_UINT32)) {
        Aipp_Kernel::AippRgbYuv<DTYPE_FEATURES, uint32_t> op;
        op.Init(images, features, tilingData);
        op.Process(tiling);
    }
}