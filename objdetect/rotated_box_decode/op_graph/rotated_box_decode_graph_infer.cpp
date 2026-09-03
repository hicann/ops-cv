/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// rotated_box_decode_package/op_graph/rotated_box_decode_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the RotatedBoxDecode operator.
//   When the Graph Engine (GE) constructs a computational graph, it needs to
//   know the output data types of each node before execution. This file
//   registers a type inference function that tells GE: "the output dtype is
//   the same as the first input's dtype."
//
//   This inference function is loaded via op_build and used by the graph
//   compiler (GE) to propagate data types through the graph. It is NOT
//   executed on the device — it's a host-side graph compilation step.
//
//   The INFER_SHAPE equivalent is in op_host/rotated_box_decode_infershape.cpp.
//
// CONTENTS:
//   - InferDataTypeForRotatedBoxDecode() — the type inference function
//   - IMPL_OP(RotatedBoxDecode).InferDataType(...) — registration macro
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : RotatedBoxDecode   — IMPL_OP macro argument, function name suffix
//   snake_case   : rotated_box_decode  — filename
//   UPPER_SNAKE  : ROTATED_BOX_DECODE  — (not used directly)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   RotatedBoxDecode → FooBar                  (IMPL_OP, function name suffix)
//   InferDataTypeForRotatedBoxDecode → InferDataTypeForFooBar
//   rotated_box_decode_graph_infer → foo_bar_graph_infer (filename)
//
// KEY MACRO: IMPL_OP(OpType).InferDataType(func)
//   Registers func as the data type inference callback for OpType.
//   The callback receives an InferDataTypeContext and can read input types
//   and set output types.
//
// =============================================================================

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration

using namespace ge;

namespace ops {
static ge::graphStatus InferDataTypeForRotatedBoxDecode(gert::InferDataTypeContext* context)
{
    // anchor_box and deltas must share the same data type; reject on mismatch.
    const ge::DataType anchor_dtype = context->GetInputDataType(0);
    const ge::DataType deltas_dtype = context->GetInputDataType(1);
    if (anchor_dtype != deltas_dtype) {
        return ge::GRAPH_FAILED;
    }
    // Output y dtype follows the first input (anchor_box).
    context->SetOutputDataType(0, anchor_dtype);
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP(RotatedBoxDecode).InferDataType(func):
//   Registers InferDataTypeForRotatedBoxDecode as the type inference function
//   for the RotatedBoxDecode operator type.
//   To create FooBar: change RotatedBoxDecode → FooBar.
IMPL_OP(RotatedBoxDecode).InferDataType(InferDataTypeForRotatedBoxDecode);
} // namespace ops
