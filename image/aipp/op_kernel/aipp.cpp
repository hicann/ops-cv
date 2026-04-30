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
 * \file aipp.cpp
 * \brief aipp kernel main
 */

#include "arch35/aipp_rgb.h"
#include "arch35/aipp_yuv.h"
#include "arch35/aipp_yuv_rgb.h"
#include "arch35/aipp_rgb_yuv.h"
#include "arch35/aipp_yuv_gray.h"
#include "arch35/aipp_rgb_gray.h"

#define AIPP_RGB_PASS_THROUGH 1
#define AIPP_YUV_PASS_THROUGH 2
#define AIPP_RGB_TO_YUV 3
#define AIPP_RGB_TO_GRAY 4
#define AIPP_YUV_TO_RGB 5
#define AIPP_YUV_TO_GRAY 6
#define AIPP_DYNAMIC_DEFAULT 100
using namespace Aipp_Kernel;

extern "C" __global__ __aicore__ void Aipp(
    GM_ADDR images, GM_ADDR params, GM_ADDR features, GM_ADDR workspace, GM_ADDR tiling)
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
    GET_PARAM_DATA_WITH_STRUCT_TBUF(tilingParam, params, TILING_KEY_IS(AIPP_DYNAMIC_DEFAULT));
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint8_t dynamicTilingKey = 0;
    AippTilingData curTilingData = tilingData;
    if (TILING_KEY_IS(AIPP_DYNAMIC_DEFAULT)) {
        UpdateRealPara(curTilingData, tilingParam.Header(), dynamicTilingKey);
        ResetDynamicTilingKey(curTilingData, dynamicTilingKey);
    }
    if (TILING_KEY_IS(AIPP_RGB_PASS_THROUGH) || dynamicTilingKey == AIPP_RGB_PASS_THROUGH) {
        Aipp_Kernel::AippRgb<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    } else if (TILING_KEY_IS(AIPP_YUV_PASS_THROUGH) || dynamicTilingKey == AIPP_YUV_PASS_THROUGH) {
        Aipp_Kernel::AippYuv<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    } else if (TILING_KEY_IS(AIPP_RGB_TO_YUV) || dynamicTilingKey == AIPP_RGB_TO_YUV) {
        Aipp_Kernel::AippRgbYuv<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    } else if (TILING_KEY_IS(AIPP_RGB_TO_GRAY) || dynamicTilingKey == AIPP_RGB_TO_GRAY) {
        Aipp_Kernel::AippRgbGray<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    } else if (TILING_KEY_IS(AIPP_YUV_TO_GRAY) || dynamicTilingKey == AIPP_YUV_TO_GRAY) {
        Aipp_Kernel::AippYuvGray<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    } else if (TILING_KEY_IS(AIPP_YUV_TO_RGB) || dynamicTilingKey == AIPP_YUV_TO_RGB) {
        Aipp_Kernel::AippYuvRgb<DTYPE_FEATURES, uint32_t> op;
        op.Init(curTilingData, tilingParam.Header(), tilingParam.GetGMParamsPtr(), dynamicTilingKey);
        op.Process(images, features);
    }
}