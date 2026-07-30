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
 * \file grid_sampler2_d_tiling_data.h
 * \brief Tiling data struct for grid_sampler2_d operator
 */

#ifndef GRID_SAMPLER2_D_TILING_DATA_H_
#define GRID_SAMPLER2_D_TILING_DATA_H_

struct GridSampler2DTilingData {
    int32_t N = 0;                 // batch size (x.shape[0])
    int32_t C = 0;                 // channel count (x.shape[1])
    int32_t H_in = 0;              // input height (x.shape[2])
    int32_t W_in = 0;              // input width (x.shape[3])
    int32_t H_out = 0;             // output height (grid.shape[1])
    int32_t W_out = 0;             // output width (grid.shape[2])
    int32_t interpolationMode = 0; // 0=bilinear, 1=nearest, 2=bicubic
    int32_t paddingMode = 0;       // 0=zeros, 1=border, 2=reflection
    int32_t alignCorners = 0;      // 0=false, 1=true
};

#endif // GRID_SAMPLER2_D_TILING_DATA_H_
