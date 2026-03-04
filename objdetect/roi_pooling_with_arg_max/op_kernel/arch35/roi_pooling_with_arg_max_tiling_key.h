/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max_tiling_key.h
 * \brief roi_pooling_with_arg_max tiling key declare
 */

#ifndef _ROI_POOLING_WITH_ARG_MAX_TILING_KEY_DECL_H_
#define _ROI_POOLING_WITH_ARG_MAX_TILING_KEY_DECL_H_

#define ROI_POOLING_WITH_ARG_MAX_TPL_FP32 0
#define ROI_POOLING_WITH_ARG_MAX_TPL_FP16 1

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    RoiPoolingWithArgMax,
    ASCENDC_TPL_DTYPE_DECL(
        dType, ROI_POOLING_WITH_ARG_MAX_TPL_FP32, ROI_POOLING_WITH_ARG_MAX_TPL_FP16) );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, ROI_POOLING_WITH_ARG_MAX_TPL_FP32) ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, ROI_POOLING_WITH_ARG_MAX_TPL_FP16) ));

#endif
