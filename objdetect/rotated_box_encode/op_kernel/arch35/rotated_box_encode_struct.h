/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_BOX_ENCODE_STRUCT_H
#define ROTATED_BOX_ENCODE_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

// DTYPE 枚举值：与 proto.md §2 dtype 组合表逐行对齐
//   ROTATED_BOX_ENCODE_DTYPE_FP16: anchor_box.dtype == float16 → fp16 升精度路径
//   ROTATED_BOX_ENCODE_DTYPE_FP32: anchor_box.dtype == float32 → fp32 直算路径
#define ROTATED_BOX_ENCODE_DTYPE_FP16 0
#define ROTATED_BOX_ENCODE_DTYPE_FP32 1

// ASCENDC_TPL_ARGS_DECL: 声明名为 DTYPE 的 uint 模板参数，默认值 1（fp32），允许值 {0, 1}
ASCENDC_TPL_ARGS_DECL(RotatedBoxEncode,
                      ASCENDC_TPL_UINT_DECL(DTYPE, 1, ASCENDC_TPL_UI_LIST, ROTATED_BOX_ENCODE_DTYPE_FP16,
                                            ROTATED_BOX_ENCODE_DTYPE_FP32));

// ASCENDC_TPL_SEL: 枚举编译期产出的 2 个特化 binary
//   TPL_SEL_0 → DTYPE=FP16（tilingKey=0）
//   TPL_SEL_1 → DTYPE=FP32（tilingKey=1）
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, ROTATED_BOX_ENCODE_DTYPE_FP16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, ROTATED_BOX_ENCODE_DTYPE_FP32)));

#endif
