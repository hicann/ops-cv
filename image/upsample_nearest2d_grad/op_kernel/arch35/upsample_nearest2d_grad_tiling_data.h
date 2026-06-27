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
 * \file upsample_nearest2d_grad_tiling_data.h
 * \brief Tiling data struct for upsample_nearest2d_grad operator
 */

#ifndef UPSAMPLE_NEAREST2D_GRAD_TILING_DATA_H_
#define UPSAMPLE_NEAREST2D_GRAD_TILING_DATA_H_

struct UpsampleNearest2dGradTilingData {
    int32_t needCoreNum = 0;
    int64_t totalElements = 0;
    int32_t dimN = 0;
    int32_t dimC = 0;
    int32_t dimHin = 0;
    int32_t dimWin = 0;
    int32_t dimHout = 0;
    int32_t dimWout = 0;
    float scaleH = 0.0f;
    float scaleW = 0.0f;
};

#endif // UPSAMPLE_NEAREST2D_GRAD_TILING_DATA_H_
