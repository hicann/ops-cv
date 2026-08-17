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
 * \file crop_and_resize_tiling_key.h
 * \brief Tiling key declare for crop_and_resize operator
 *
 * Single template parameter:
 *   schMode (UINT 1-bit): scene mode
 *     0 = CROP_AND_RESIZE_MODE_BILINEAR (bilinear interpolation)
 *
 * dtype is NOT encoded in TilingKey; DTYPE_ macros auto-instantiate all dtype combinations.
 */

#ifndef CROP_AND_RESIZE_TILING_KEY_H_
#define CROP_AND_RESIZE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// 单一场景模式：bilinear 插值（method 属性固定为 bilinear）
// TilingKey 仅编码场景模式，禁止枚举 dtype
#define CROP_AND_RESIZE_MODE_BILINEAR 0

ASCENDC_TPL_ARGS_DECL(CropAndResize,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_BILINEAR));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_BILINEAR),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(CropAndResizeTilingData)));

#endif // CROP_AND_RESIZE_TILING_KEY_H_
