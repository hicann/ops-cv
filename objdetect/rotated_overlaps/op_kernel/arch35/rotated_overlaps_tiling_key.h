/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_OVERLAPS_TILING_KEY_H_
#define ROTATED_OVERLAPS_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define ROTATED_OVERLAPS_TPL_XYWHT 0U
#define ROTATED_OVERLAPS_TPL_XYXYT 1U
#define ROTATED_OVERLAPS_TPL_INDEX_64 0U
#define ROTATED_OVERLAPS_TPL_INDEX_32 1U

ASCENDC_TPL_ARGS_DECL(RotatedOverlaps,
                      ASCENDC_TPL_UINT_DECL(trans, 1, ASCENDC_TPL_UI_LIST, ROTATED_OVERLAPS_TPL_XYWHT,
                                            ROTATED_OVERLAPS_TPL_XYXYT),
                      ASCENDC_TPL_UINT_DECL(use32Bit, 1, ASCENDC_TPL_UI_LIST, ROTATED_OVERLAPS_TPL_INDEX_64,
                                            ROTATED_OVERLAPS_TPL_INDEX_32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ASCENDC_TPL_UINT_SEL(trans, ASCENDC_TPL_UI_LIST, ROTATED_OVERLAPS_TPL_XYWHT, ROTATED_OVERLAPS_TPL_XYXYT),
    ASCENDC_TPL_UINT_SEL(use32Bit, ASCENDC_TPL_UI_LIST, ROTATED_OVERLAPS_TPL_INDEX_64, ROTATED_OVERLAPS_TPL_INDEX_32),
    ASCENDC_TPL_TILING_STRUCT_SEL(RotatedOverlapsTilingData)));

#endif // ROTATED_OVERLAPS_TILING_KEY_H_
