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
 * \file upsample_nearest3d_tiling_key.h
 * \brief upsample_nearest3d tiling key declare
 */

#ifndef _UPSAMPLE_NEAREST3D_TILING_KEY_DECL_H_
#define _UPSAMPLE_NEAREST3D_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define SCH_MODE_0 0 // 纯copy模板
#define SCH_MODE_1 1 // dhw 分线程， nc拉成一维
#define SCH_MODE_2 2 // cdhw 分线程, 和竞品一致
#define SCH_MODE_3 3 // ncdhw 分线程

#define UPSAMPLE_NEAREST3D_TPL_KEY_DECL()                          \
    ASCENDC_TPL_UINT_DECL(schId,                                   \
        ASCENDC_TPL_8_BW,                                          \
        ASCENDC_TPL_UI_LIST,                                       \
        SCH_MODE_0,                                                \
        SCH_MODE_1,                                                \
        SCH_MODE_2,                                                \
        SCH_MODE_3),                                               \
    ASCENDC_TPL_UINT_DECL(isUint32,                                \
        ASCENDC_TPL_8_BW,                                          \
        ASCENDC_TPL_UI_LIST,                                       \
        0,                                                         \
        1)

#define UPSAMPLE_NEAREST3D_COPY_TPL_KEY_SEL()                      \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SCH_MODE_0),  \
        ASCENDC_TPL_UINT_SEL(isUint32, ASCENDC_TPL_UI_LIST, 0)     \

#define UPSAMPLE_NEAREST3D_ELSE_TPL_KEY_SEL()                      \
    ASCENDC_TPL_UINT_SEL(schId,                                    \
        ASCENDC_TPL_UI_LIST,                                       \
        SCH_MODE_1,                                                \
        SCH_MODE_2,                                                \
        SCH_MODE_3),                                               \
        ASCENDC_TPL_UINT_SEL(isUint32, ASCENDC_TPL_UI_LIST, 0, 1)

ASCENDC_TPL_ARGS_DECL(UpsampleNearest3d, UPSAMPLE_NEAREST3D_TPL_KEY_DECL());
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(UPSAMPLE_NEAREST3D_COPY_TPL_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(UPSAMPLE_NEAREST3D_ELSE_TPL_KEY_SEL()));

#endif
