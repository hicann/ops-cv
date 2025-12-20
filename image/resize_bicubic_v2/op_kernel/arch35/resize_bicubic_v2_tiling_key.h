/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_bicubic_v2_tiling_key.h
 * \brief resize_bicubic_v2 tiling key declare
 */

#ifndef _RESIZE_BICUBIC_V2_TILING_KEY_DECL_H_
#define _RESIZE_BICUBIC_V2_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_0 0
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_1 1
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_2 2
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_3 3
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_4 4
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_5 5
#define RESIZE_BICUBIC_V2_TPL_SCH_MODE_6 6
#define RESIZE_BICUBIC_V2_TPL_INT32_0 0
#define RESIZE_BICUBIC_V2_TPL_INT32_1 1
#define RESIZE_BICUBIC_V2_TPL_HALF_0 0
#define RESIZE_BICUBIC_V2_TPL_HALF_1 1
#define RESIZE_BICUBIC_V2_TPL_NCHW_0 0
#define RESIZE_BICUBIC_V2_TPL_NCHW_1 1

#define RESIZE_BICUBIC_V2_TPL_KEY_DECL()    \
    ASCENDC_TPL_UINT_DECL(schId,            \
        ASCENDC_TPL_8_BW,                   \
        ASCENDC_TPL_UI_LIST,                \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_0,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_1,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_2,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_3,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_4,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_5,   \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_6),  \
        ASCENDC_TPL_UINT_DECL(isInt32,      \
            ASCENDC_TPL_8_BW,               \
            ASCENDC_TPL_UI_LIST,            \
            RESIZE_BICUBIC_V2_TPL_INT32_0,  \
            RESIZE_BICUBIC_V2_TPL_INT32_1), \
        ASCENDC_TPL_UINT_DECL(isHalfPixel,  \
            ASCENDC_TPL_8_BW,               \
            ASCENDC_TPL_UI_LIST,            \
            RESIZE_BICUBIC_V2_TPL_HALF_0,   \
            RESIZE_BICUBIC_V2_TPL_HALF_1),  \
        ASCENDC_TPL_UINT_DECL(              \
            isNchw, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, RESIZE_BICUBIC_V2_TPL_NCHW_0, RESIZE_BICUBIC_V2_TPL_NCHW_1)

#define RESIZE_BICUBIC_V2_NCHW_TPL_KEY_SEL()                                                   \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_RANGE, 1, 0, RESIZE_BICUBIC_V2_TPL_SCH_MODE_4), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_RANGE, 1, 0, 1),                          \
        ASCENDC_TPL_UINT_SEL(isHalfPixel, ASCENDC_TPL_UI_RANGE, 1, 0, 1),                      \
        ASCENDC_TPL_UINT_SEL(isNchw, ASCENDC_TPL_UI_LIST, RESIZE_BICUBIC_V2_TPL_NCHW_1)

#define RESIZE_BICUBIC_V2_NHWC_TPL_KEY_SEL()                              \
    ASCENDC_TPL_UINT_SEL(schId,                                           \
        ASCENDC_TPL_UI_LIST,                                              \
        0,                                                                \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_2,                                 \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_3,                                 \
        RESIZE_BICUBIC_V2_TPL_SCH_MODE_4),                                \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_RANGE, 1, 0, 1),     \
        ASCENDC_TPL_UINT_SEL(isHalfPixel, ASCENDC_TPL_UI_RANGE, 1, 0, 1), \
        ASCENDC_TPL_UINT_SEL(isNchw, ASCENDC_TPL_UI_LIST, 0)

#define RESIZE_BICUBIC_V2_SIMD_TPL_KEY_SEL()                                                             \
    ASCENDC_TPL_UINT_SEL(                                                                                \
        schId, ASCENDC_TPL_UI_LIST, RESIZE_BICUBIC_V2_TPL_SCH_MODE_5, RESIZE_BICUBIC_V2_TPL_SCH_MODE_6), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0),                                           \
        ASCENDC_TPL_UINT_SEL(isHalfPixel, ASCENDC_TPL_UI_LIST, 0, 1),                                    \
        ASCENDC_TPL_UINT_SEL(isNchw, ASCENDC_TPL_UI_LIST, 0)

ASCENDC_TPL_ARGS_DECL(ResizeBicubicV2, RESIZE_BICUBIC_V2_TPL_KEY_DECL());
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RESIZE_BICUBIC_V2_NCHW_TPL_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(RESIZE_BICUBIC_V2_NHWC_TPL_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(RESIZE_BICUBIC_V2_SIMD_TPL_KEY_SEL()));

#endif
