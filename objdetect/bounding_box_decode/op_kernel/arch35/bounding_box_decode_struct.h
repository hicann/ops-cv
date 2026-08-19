/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BOUNDING_BOX_DECODE_STRUCT_H
#define BOUNDING_BOX_DECODE_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// DataType numeric constants (match ge::DataType enum values)
#define BOUNDING_BOX_DECODE_TPL_FP32 0 // ge::DT_FLOAT
#define BOUNDING_BOX_DECODE_TPL_FP16 1 // ge::DT_FLOAT16

// ---------------------------------------------------------------------------
// TPL parameter declarations
//   T : DATATYPE, selectable from {FP16, FP32}
//
// NOTE: ASCENDC_TPL_DATATYPE_* (not ASCENDC_TPL_DTYPE_*) is used so that
//   the codegen emits TypeFromId<N>::type as the template argument, which
//   resolves to the C++ type (half / float) and matches the
//   `template <typename T>` kernel signature in proto.md.
//   ASCENDC_TPL_DTYPE_* would emit raw integers, only matching `int T`.
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(BoundingBoxDecode,
                      ASCENDC_TPL_DATATYPE_DECL(T, BOUNDING_BOX_DECODE_TPL_FP16, BOUNDING_BOX_DECODE_TPL_FP32));

// ---------------------------------------------------------------------------
// TPL specialisations — tilingKey = T_value
//   (T=FP32) → tilingKey = 0   (fp32, handles both normal and empty at runtime)
//   (T=FP16) → tilingKey = 1   (fp16, handles both normal and empty at runtime)
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(T, BOUNDING_BOX_DECODE_TPL_FP16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(T, BOUNDING_BOX_DECODE_TPL_FP32)));

#endif
