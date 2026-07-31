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
 * \file crop_tiling_key.h
 * \brief Tiling key declaration for crop operator
 *
 * Single template parameter:
 *   idxWidth (UINT 1-bit): index width
 *     CROP_IDX_32 = 0  -> 32-bit index (uint32_t)
 *     CROP_IDX_64 = 1  -> 64-bit index (uint64_t)
 *   dtype is NOT encoded; handled by DTYPE_X macro.
 */

#ifndef CROP_TILING_KEY_H_
#define CROP_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define CROP_IDX_32 0 // 32-bit index
#define CROP_IDX_64 1 // 64-bit index

ASCENDC_TPL_ARGS_DECL(Crop, ASCENDC_TPL_UINT_DECL(idxWidth, 1, ASCENDC_TPL_UI_LIST, CROP_IDX_32, CROP_IDX_64));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(idxWidth, ASCENDC_TPL_UI_LIST, CROP_IDX_32),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(CropTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(idxWidth, ASCENDC_TPL_UI_LIST, CROP_IDX_64),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(CropTilingData)));

#endif // CROP_TILING_KEY_H_
