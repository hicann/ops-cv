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
 * Single scene mode (NORMAL=0). Dtype is NOT encoded in TilingKey;
 * it is handled by DTYPE_ macro auto-instantiation (MDE §3.2).
 */

#ifndef DILATION2D_BACKPROP_FILTER_TILING_KEY_H_
#define DILATION2D_BACKPROP_FILTER_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define DILATION2D_BACKPROP_FILTER_MODE_NORMAL 0

ASCENDC_TPL_ARGS_DECL(Dilation2DBackpropFilter,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, DILATION2D_BACKPROP_FILTER_MODE_NORMAL));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          DILATION2D_BACKPROP_FILTER_MODE_NORMAL),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(Dilation2DBackpropFilterTilingData)));

#endif // DILATION2D_BACKPROP_FILTER_TILING_KEY_H_
