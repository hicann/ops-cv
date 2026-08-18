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
 * \file points_in_polygons_tiling_key.h
 * \brief PointsInPolygons TilingKey definition and ASCENDC_TPL declaration
 */

#ifndef POINTS_IN_POLYGONS_TILING_KEY_H_
#define POINTS_IN_POLYGONS_TILING_KEY_H_
#include "ascendc/host_api/tiling/template_argument.h"

// 用 #define 而非 constexpr：AscendC codegen 在预处理阶段文本解析 ASCENDC_TPL 占位符值列表，
// 须为预处理器可解析的字面量；constexpr 是 C++ 语义，codegen 会退化为单份 sub-kernel。
#define POINTS_IN_POLYGONS_KEY_EMPTY 0
#define POINTS_IN_POLYGONS_KEY_NORMAL 1
#define POINTS_IN_POLYGONS_KEY_N_VEC 2

ASCENDC_TPL_ARGS_DECL(PointsInPolygons,
                      ASCENDC_TPL_UINT_DECL(KEY, 3, ASCENDC_TPL_UI_LIST, POINTS_IN_POLYGONS_KEY_EMPTY,
                                            POINTS_IN_POLYGONS_KEY_NORMAL, POINTS_IN_POLYGONS_KEY_N_VEC));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, POINTS_IN_POLYGONS_KEY_EMPTY)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, POINTS_IN_POLYGONS_KEY_NORMAL)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, POINTS_IN_POLYGONS_KEY_N_VEC)));

#endif // POINTS_IN_POLYGONS_TILING_KEY_H_
