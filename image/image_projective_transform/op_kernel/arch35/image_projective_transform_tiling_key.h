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
 * \file image_projective_transform_tiling_key.h
 * \brief Tiling key declaration for image_projective_transform operator
 *
 * One template parameter:
 *   interpMode (UINT 1-bit): interpolation mode
 *     0 = BILINEAR, 1 = NEAREST
 *
 * dtype is handled by DTYPE_IMAGES macro, not enumerated here.
 */

#ifndef IMAGE_PROJECTIVE_TRANSFORM_TILING_KEY_H_
#define IMAGE_PROJECTIVE_TRANSFORM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define IPT_TPL_BILINEAR 0
#define IPT_TPL_NEAREST 1

ASCENDC_TPL_ARGS_DECL(ImageProjectiveTransform,
                      ASCENDC_TPL_UINT_DECL(interpMode, 1, ASCENDC_TPL_UI_LIST, IPT_TPL_BILINEAR, IPT_TPL_NEAREST));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(interpMode, ASCENDC_TPL_UI_LIST, IPT_TPL_BILINEAR),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(ImageProjectiveTransformTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(interpMode, ASCENDC_TPL_UI_LIST, IPT_TPL_NEAREST),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(ImageProjectiveTransformTilingData)));

#endif // IMAGE_PROJECTIVE_TRANSFORM_TILING_KEY_H_
