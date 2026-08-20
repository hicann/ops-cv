/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_TILING_KEY_H_
#define GAUSSIAN_BLUR_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define GAUSSIAN_BLUR_PASS_ROW_W128 0
#define GAUSSIAN_BLUR_PASS_COLUMN_H96 1
#define GAUSSIAN_BLUR_PASS_ROW_W192 2
#define GAUSSIAN_BLUR_PASS_COLUMN_H64 3
#define GAUSSIAN_BLUR_PASS_COLUMN_H128 4
#define GAUSSIAN_BLUR_PASS_ROW_W96 5
#define GAUSSIAN_BLUR_PASS_FUSED_GENERIC_C8 6
#define GAUSSIAN_BLUR_PASS_FUSED_K31_C4_RING 7

// Compile only the production row, column, and fused tiling keys. This keeps
// package build time bounded while retaining every runtime delivery path.
#ifndef GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
#define GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS 1
#endif

// K31 ring isolation stage: 0=full ring, 1=external-UB/512-thread marker,
// 2=runtime-loop horizontal initialization plus a 31-row sample dump,
// 3=horizontal initialization plus the complete first output row;
// 4=dump VF arguments and both Gaussian weight vectors.
#ifndef GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE
#define GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE 0
#endif

// Keep compile-time specialization on the hot K3/K5/K11/K21 set. Other
// kernel sizes use the runtime KernelSize=0 implementation to limit OPC time.
#ifndef GAUSSIAN_BLUR_ENABLE_COLUMN_HOT_K_SPECIALIZATION
#define GAUSSIAN_BLUR_ENABLE_COLUMN_HOT_K_SPECIALIZATION 1
#endif

#ifndef GAUSSIAN_BLUR_ENABLE_ROW_HOT_K_SPECIALIZATION
#define GAUSSIAN_BLUR_ENABLE_ROW_HOT_K_SPECIALIZATION 1
#endif

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
ASCENDC_TPL_ARGS_DECL(GaussianBlur,
                      ASCENDC_TPL_UINT_DECL(passVariant, 4, ASCENDC_TPL_UI_LIST, GAUSSIAN_BLUR_PASS_ROW_W128,
                                            GAUSSIAN_BLUR_PASS_COLUMN_H96, GAUSSIAN_BLUR_PASS_FUSED_K31_C4_RING));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(passVariant, ASCENDC_TPL_UI_LIST, GAUSSIAN_BLUR_PASS_ROW_W128,
                                                          GAUSSIAN_BLUR_PASS_COLUMN_H96,
                                                          GAUSSIAN_BLUR_PASS_FUSED_K31_C4_RING)));
#else
ASCENDC_TPL_ARGS_DECL(GaussianBlur,
                      ASCENDC_TPL_UINT_DECL(passVariant, 4, ASCENDC_TPL_UI_LIST, GAUSSIAN_BLUR_PASS_ROW_W128,
                                            GAUSSIAN_BLUR_PASS_COLUMN_H96, GAUSSIAN_BLUR_PASS_FUSED_GENERIC_C8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(passVariant, ASCENDC_TPL_UI_LIST, GAUSSIAN_BLUR_PASS_ROW_W128,
                                                          GAUSSIAN_BLUR_PASS_COLUMN_H96,
                                                          GAUSSIAN_BLUR_PASS_FUSED_GENERIC_C8)));
#endif

#endif // GAUSSIAN_BLUR_TILING_KEY_H_
