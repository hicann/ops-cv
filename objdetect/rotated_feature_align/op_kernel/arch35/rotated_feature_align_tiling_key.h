/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_FEATURE_ALIGN_TILING_KEY_H_
#define ROTATED_FEATURE_ALIGN_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define RFA_POINTS_MODE_1 0
#define RFA_POINTS_MODE_5 1

ASCENDC_TPL_ARGS_DECL(RotatedFeatureAlign,
                      ASCENDC_TPL_UINT_DECL(pointsMode, 1, ASCENDC_TPL_UI_LIST, RFA_POINTS_MODE_1, RFA_POINTS_MODE_5));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(pointsMode, ASCENDC_TPL_UI_LIST, RFA_POINTS_MODE_1,
                                                          RFA_POINTS_MODE_5)));

#endif // ROTATED_FEATURE_ALIGN_TILING_KEY_H_
