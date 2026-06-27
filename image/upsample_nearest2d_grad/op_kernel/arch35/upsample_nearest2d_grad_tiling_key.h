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
 * \file upsample_nearest2d_grad_tiling_key.h
 * \brief Tiling key declaration for upsample_nearest2d_grad operator
 *
 * Single template parameter:
 *   schMode (UINT 1-bit): scene mode (only MODE_DEFAULT = 0)
 *   dtype is handled by DTYPE_GRAD_OUTPUT macro automatically
 */

#ifndef UPSAMPLE_NEAREST2D_GRAD_TILING_KEY_H_
#define UPSAMPLE_NEAREST2D_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define UPSAMPLE_NEAREST2D_GRAD_MODE_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(UpsampleNearest2dGrad,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, UPSAMPLE_NEAREST2D_GRAD_MODE_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          UPSAMPLE_NEAREST2D_GRAD_MODE_DEFAULT),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(UpsampleNearest2dGradTilingData)));

#endif // UPSAMPLE_NEAREST2D_GRAD_TILING_KEY_H_
