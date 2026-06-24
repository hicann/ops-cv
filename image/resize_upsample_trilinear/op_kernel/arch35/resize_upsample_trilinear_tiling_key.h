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
 * \file resize_upsample_trilinear_tiling_key.h
 * \brief resize_upsample_trilinear tiling key declare for arch35
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_ARCH35_TILING_KEY_H
#define RESIZE_UPSAMPLE_TRILINEAR_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_DTYPE_FP32 0
#define TPL_DTYPE_FP16 1
#define TPL_DTYPE_BF16 2

ASCENDC_TPL_ARGS_DECL(ResizeUpsampleTrilinear, ASCENDC_TPL_UINT_DECL(dtypeKey, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                                                                     TPL_DTYPE_FP32, TPL_DTYPE_FP16, TPL_DTYPE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(dtypeKey, ASCENDC_TPL_UI_RANGE, 1, TPL_DTYPE_FP32,
                                                          TPL_DTYPE_BF16)));

#endif // RESIZE_UPSAMPLE_TRILINEAR_ARCH35_TILING_KEY_H