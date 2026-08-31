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
 * \file deformable_roi_pool_tiling_key.h
 * \brief Tiling key declare for deformable_roi_pool operator
 */

#ifndef DEFORMABLE_ROI_POOL_TILING_KEY_H_
#define DEFORMABLE_ROI_POOL_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define DEFORMABLE_ROI_POOL_TPL_WITH_OFFSET 0
#define DEFORMABLE_ROI_POOL_TPL_NO_OFFSET 1

ASCENDC_TPL_ARGS_DECL(DeformableRoiPool,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, DEFORMABLE_ROI_POOL_TPL_WITH_OFFSET,
                                            DEFORMABLE_ROI_POOL_TPL_NO_OFFSET));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, DEFORMABLE_ROI_POOL_TPL_WITH_OFFSET),
                         ASCENDC_TPL_TILING_STRUCT_SEL(DeformableRoiPoolTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, DEFORMABLE_ROI_POOL_TPL_NO_OFFSET),
                         ASCENDC_TPL_TILING_STRUCT_SEL(DeformableRoiPoolTilingData)));

#endif // DEFORMABLE_ROI_POOL_TILING_KEY_H_
