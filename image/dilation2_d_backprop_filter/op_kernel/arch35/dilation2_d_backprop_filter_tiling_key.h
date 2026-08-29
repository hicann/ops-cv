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
 * \file dilation2_d_backprop_filter_tiling_key.h
 * \brief Tiling key declaration for dilation2_d_backprop_filter operator
 *
 * Two TilingKey parameters (detMode + schMode):
 *   detMode: 0=DETERMINISTIC, 1=NON_DETERMINISTIC
 *   schMode: 0=NORMAL (only value, reserved for future)
 *
 * 2 kernel variants:
 *   #1 detMode=0(DET),     schMode=0(NORMAL) — deterministic (Dilation2DBackpropFilterTilingData)
 *   #2 detMode=1(NON_DET), schMode=0(NORMAL) — non-deterministic (Dilation2DBackpropFilterNonDetTilingData)
 *
 * Dtype is NOT encoded in TilingKey; handled by DTYPE_ macro auto-instantiation.
 */

#ifndef DILATION2D_BACKPROP_FILTER_TILING_KEY_H_
#define DILATION2D_BACKPROP_FILTER_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define DILATION2D_BACKPROP_FILTER_MODE_NORMAL 0
#define DILATION2D_BACKPROP_FILTER_MODE_DETERMINISTIC 0
#define DILATION2D_BACKPROP_FILTER_MODE_NON_DETERMINISTIC 1

ASCENDC_TPL_ARGS_DECL(Dilation2DBackpropFilter,
                      ASCENDC_TPL_UINT_DECL(detMode, 1, ASCENDC_TPL_UI_LIST,
                                            DILATION2D_BACKPROP_FILTER_MODE_DETERMINISTIC,
                                            DILATION2D_BACKPROP_FILTER_MODE_NON_DETERMINISTIC),
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, DILATION2D_BACKPROP_FILTER_MODE_NORMAL));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ASCENDC_TPL_UINT_SEL(detMode, ASCENDC_TPL_UI_LIST, DILATION2D_BACKPROP_FILTER_MODE_DETERMINISTIC,
                         DILATION2D_BACKPROP_FILTER_MODE_NON_DETERMINISTIC),
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, DILATION2D_BACKPROP_FILTER_MODE_NORMAL),
    ASCENDC_TPL_TILING_STRUCT_SEL(Dilation2DBackpropFilterTilingData)));

#endif // DILATION2D_BACKPROP_FILTER_TILING_KEY_H_
