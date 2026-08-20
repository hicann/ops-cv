/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_UT_TILING_H_
#define GAUSSIAN_BLUR_UT_TILING_H_

#include <cstring>
#include "../../../op_kernel/arch35/gaussian_blur_tiling_data.h"

#define __aicore__

inline void InitTilingData(uint8_t* tiling, GaussianBlurTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(GaussianBlurTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif // GAUSSIAN_BLUR_UT_TILING_H_
