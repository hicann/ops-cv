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
 * \file dilation2_d_backprop_input_tiling_key.h
 * \brief Tiling key declaration for dilation2_d_backprop_input operator
 *
 * Two TilingKey parameters (schMode + deterministic):
 *   schMode: 0=NHWC, 1=NCHW
 *   deterministic: 0=NO (non-deterministic), 1=YES (deterministic)
 *
 * Final tilingKey value is bit-packed as (deterministic << 1) | schMode:
 *   key=0: schMode=NHWC, deterministic=NO  — non-deterministic NHWC (zero+atomic_add)
 *   key=1: schMode=NCHW, deterministic=NO  — non-deterministic NCHW (zero+atomic_add)
 *   key=2: schMode=NHWC, deterministic=YES — deterministic NHWC (two-phase argmax+accumulate)
 *   key=3: schMode=NCHW, deterministic=YES — deterministic NCHW (two-phase argmax+accumulate)
 */

#ifndef DILATION2_D_BACKPROP_INPUT_TILING_KEY_H_
#define DILATION2_D_BACKPROP_INPUT_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define DILATION2_D_BACKPROP_INPUT_SCH_MODE_NHWC 0
#define DILATION2_D_BACKPROP_INPUT_SCH_MODE_NCHW 1
#define DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_NO 0
#define DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_YES 1

ASCENDC_TPL_ARGS_DECL(Dilation2DBackpropInput,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, DILATION2_D_BACKPROP_INPUT_SCH_MODE_NHWC,
                                            DILATION2_D_BACKPROP_INPUT_SCH_MODE_NCHW),
                      ASCENDC_TPL_UINT_DECL(deterministic, 1, ASCENDC_TPL_UI_LIST,
                                            DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_NO,
                                            DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_YES));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          DILATION2_D_BACKPROP_INPUT_SCH_MODE_NHWC,
                                                          DILATION2_D_BACKPROP_INPUT_SCH_MODE_NCHW),
                                     ASCENDC_TPL_UINT_SEL(deterministic, ASCENDC_TPL_UI_LIST,
                                                          DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_NO,
                                                          DILATION2_D_BACKPROP_INPUT_DETERMINISTIC_YES),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(Dilation2DBackpropInputTilingData)));

#endif // DILATION2_D_BACKPROP_INPUT_TILING_KEY_H_
