/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file col2im_tiling_key.h
 * \brief col2im tiling key declare
 */

#ifndef _COL2IM_TILING_KEY_DECL_H_
#define _COL2IM_TILING_KEY_DECL_H_

#define COL2IM_TPL_FP32 0
#define COL2IM_TPL_FP16 1
#define COL2IM_TPL_BF16 2

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Col2im,
    ASCENDC_TPL_DTYPE_DECL(
        dType, COL2IM_TPL_FP32, COL2IM_TPL_FP16, COL2IM_TPL_BF16) );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, COL2IM_TPL_FP32) ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, COL2IM_TPL_FP16) ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, COL2IM_TPL_BF16) ));

#endif
