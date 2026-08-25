/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DECODE_BBOX_V2_STRUCT_H
#define DECODE_BBOX_V2_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define DECODE_BBOX_V2_LAYOUT_N4 0
#define DECODE_BBOX_V2_LAYOUT_F4N 1

ASCENDC_TPL_ARGS_DECL(DecodeBboxV2, ASCENDC_TPL_UINT_DECL(LAYOUT, 1, ASCENDC_TPL_UI_LIST, DECODE_BBOX_V2_LAYOUT_N4,
                                                          DECODE_BBOX_V2_LAYOUT_F4N));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, DECODE_BBOX_V2_LAYOUT_N4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, DECODE_BBOX_V2_LAYOUT_F4N)));

#endif
