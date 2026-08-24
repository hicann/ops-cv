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
 * \file stack_group_points_tiling_key.h
 * \brief Tiling key declare for stack_group_points operator
 *
 */

#ifndef STACK_GROUP_POINTS_TILING_KEY_H_
#define STACK_GROUP_POINTS_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define STACK_GROUP_POINTS_TPL_FP32 0
#define STACK_GROUP_POINTS_TPL_FP16 1

ASCENDC_TPL_ARGS_DECL(StackGroupPoints,
                      ASCENDC_TPL_DTYPE_DECL(schMode, STACK_GROUP_POINTS_TPL_FP32, STACK_GROUP_POINTS_TPL_FP16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(schMode, STACK_GROUP_POINTS_TPL_FP32)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(schMode, STACK_GROUP_POINTS_TPL_FP16)));

#endif // STACK_GROUP_POINTS_TILING_KEY_H_
