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
 * \file grid_sampler2_d_tiling_key.h
 * \brief Tiling key declaration for grid_sampler2_d operator
 *
 * Template parameters:
 *   interpMode (UINT 2-bit): interpolation mode
 *     0 = BILINEAR, 1 = NEAREST, 2 = BICUBIC
 *
 * dtype is handled by DTYPE_X macro (auto-instantiated per dtype combination).
 * Single scene mode — padding/align_corners are runtime parameters.
 */

#ifndef GRID_SAMPLER2_D_TILING_KEY_H_
#define GRID_SAMPLER2_D_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define GRID_SAMPLER_2D_BILINEAR 0
#define GRID_SAMPLER_2D_NEAREST 1
#define GRID_SAMPLER_2D_BICUBIC 2

ASCENDC_TPL_ARGS_DECL(GridSampler2D, ASCENDC_TPL_UINT_DECL(interpMode, 2, ASCENDC_TPL_UI_LIST, GRID_SAMPLER_2D_BILINEAR,
                                                           GRID_SAMPLER_2D_NEAREST, GRID_SAMPLER_2D_BICUBIC));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(interpMode, ASCENDC_TPL_UI_LIST, GRID_SAMPLER_2D_BILINEAR),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GridSampler2DTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(interpMode, ASCENDC_TPL_UI_LIST, GRID_SAMPLER_2D_NEAREST),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GridSampler2DTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(interpMode, ASCENDC_TPL_UI_LIST, GRID_SAMPLER_2D_BICUBIC),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GridSampler2DTilingData)));

#endif // GRID_SAMPLER2_D_TILING_KEY_H_
