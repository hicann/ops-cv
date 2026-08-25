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
// rotated_box_decode_package/op_host/rotated_box_decode_infershape.cpp
// =============================================================================
//
// ROLE: Shape inference for the RotatedBoxDecode operator.
//   When the Graph Engine compiles a graph with dynamic shapes (or when
//   shape propagation is needed), it calls the infer-shape function to
//   determine the output shape from the input shapes.
//
//   For RotatedBoxDecode (y = x * scale + bias), the output shape is identical
//   to the input x's shape because scale and bias are broadcast to x.
//   This is the simplest case of shape inference — a direct copy.
//
// CONTENTS:
//   - InferShape4RotatedBoxDecode() — copies input shape to output shape
//   - IMPL_OP_INFERSHAPE(RotatedBoxDecode).InferShape(...) — registration macro
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : RotatedBoxDecode   — IMPL_OP_INFERSHAPE argument, function name suffix
//   snake_case   : rotated_box_decode  — filename
//   UPPER_SNAKE  : ROTATED_BOX_DECODE  — (not used directly)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   RotatedBoxDecode → FooBar                           (IMPL_OP_INFERSHAPE, function suffix)
//   InferShape4RotatedBoxDecode → InferShape4FooBar     (function name)
//   rotated_box_decode_infershape → foo_bar_infershape   (filename)
//
//   For operators that change shape (e.g., reshape, concat), the infer shape
//   function would compute the output shape based on input shapes and attributes.
//   For element-wise ops like RotatedBoxDecode, output shape = input shape.
//
// =============================================================================

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro
#include "graph/operator_reg.h"                    // COMMON_INFER_FUNC_REG (V1 InferShape)

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// IsUnknownRank(shape) — true when shape represents an unknown-rank tensor.
//
// In CANN, ge::UNKNOWN_RANK is {-2}: a single-dimension shape whose only
// value is -2.  Such shapes cannot be validated against another shape, so
// the caller must skip compatibility checks for them.
// ---------------------------------------------------------------------------
static bool IsUnknownRank(const gert::Shape& shape) { return shape.GetDimNum() == 1U && shape.GetDim(0) == -2; }

// ---------------------------------------------------------------------------
// ShapesCompatible(a, b) — true when two shapes are broadcast-compatible for
// RotatedBoxDecode (which requires anchor_box and deltas to share the same
// shape).
//
// Rules:
//   - If either shape is unknown-rank ({-2}), treat as compatible (cannot
//     validate).
//   - If the ranks differ, the shapes are incompatible.
//   - For each dimension: reject only when both values are known (>= 0) and
//     differ; unknown dims (-1) are tolerated.
// ---------------------------------------------------------------------------
static bool ShapesCompatible(const gert::Shape& a, const gert::Shape& b)
{
    if (IsUnknownRank(a) || IsUnknownRank(b)) {
        return true;
    }
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        const int64_t da = a.GetDim(i);
        const int64_t db = b.GetDim(i);
        if (da >= 0 && db >= 0 && da != db) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// InferShape4RotatedBoxDecode(context) — shape inference function
//
// Validates that anchor_box (input 0) and deltas (input 1) have compatible
// shapes, then copies the anchor_box shape to output y.  When the shapes
// conflict (e.g. [2,5,100] vs [2,5,50]), GRAPH_FAILED is returned so that
// the graph compiler rejects the graph.
//
// Parameters:
//   context — InferShapeContext providing access to input shapes and
//             allowing setting of output shapes
//
// Returns:
//   ge::GRAPH_SUCCESS on success, ge::GRAPH_FAILED on shape mismatch
//
// To create FooBar: rename to InferShape4FooBar.
//   For element-wise ops, this copy logic works unchanged.
//   For shape-changing ops, compute output shape from input shapes + attrs.
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4RotatedBoxDecode(gert::InferShapeContext* context)
{
    // GetInputShape(0): reads the shape of the first input (index 0 = anchor_box)
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    // Validate deltas (input 1) against anchor_box when available.
    const gert::Shape* deltas_shape = context->GetInputShape(1);
    if (deltas_shape != nullptr && !ShapesCompatible(*input_shape, *deltas_shape)) {
        return ge::GRAPH_FAILED;
    }

    // GetOutputShape(0): gets a mutable reference to output y's shape descriptor
    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // Copy: output shape = input shape (element-wise, scale/bias are broadcast)
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(RotatedBoxDecode).InferShape(func):
//   Registers InferShape4RotatedBoxDecode as the shape inference function
//   for the RotatedBoxDecode operator type.
//   To create FooBar: change RotatedBoxDecode → FooBar.
IMPL_OP_INFERSHAPE(RotatedBoxDecode).InferShape(InferShape4RotatedBoxDecode);

// ---------------------------------------------------------------------------
// V1 InferShapeAndType — legacy ge::Operator-based shape inference.
//
// The V1 path (ge::Operator::InferShapeAndType) does NOT bridge to the V2
// gert callback above.  When no V1 InferShapeFunc is registered, the ge
// framework falls back to a default that copies input[0]'s shape to the
// output but always returns GRAPH_SUCCESS, making it impossible to reject
// mismatched shapes.  Registering this V1 function gives the V1 path the
// same shape-compatibility validation as the V2 path.
//
// Signature: graphStatus func(ge::Operator &op)  (per COMMON_INFER_FUNC_REG)
// ---------------------------------------------------------------------------
static ge::graphStatus InferShapeAndTypeForRotatedBoxDecode(ge::Operator& op)
{
    const ge::TensorDesc anchor_desc = op.GetInputDesc("anchor_box");
    const ge::TensorDesc deltas_desc = op.GetInputDesc("deltas");
    const ge::Shape anchor_shape = anchor_desc.GetShape();
    const ge::Shape deltas_shape = deltas_desc.GetShape();

    // Reject when both inputs have known, differing shapes.
    // Unknown-rank shapes ({-2}) cannot be validated and are accepted.
    const bool anchor_unknown = (anchor_shape.GetDims() == ge::UNKNOWN_RANK);
    const bool deltas_unknown = (deltas_shape.GetDims() == ge::UNKNOWN_RANK);
    if (!anchor_unknown && !deltas_unknown && anchor_shape.GetDims() != deltas_shape.GetDims()) {
        return ge::GRAPH_FAILED;
    }

    // Output y: same shape and dtype as anchor_box.
    ge::TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(anchor_shape);
    output_desc.SetOriginShape(anchor_shape);
    output_desc.SetDataType(anchor_desc.GetDataType());
    op.UpdateOutputDesc("y", output_desc);

    return ge::GRAPH_SUCCESS;
}

// COMMON_INFER_FUNC_REG registers a V1 InferShapeFunc for ge::Operator::
// InferShapeAndType().  To create FooBar: change RotatedBoxDecode → FooBar.
COMMON_INFER_FUNC_REG(RotatedBoxDecode, InferShapeAndTypeForRotatedBoxDecode);

} // namespace ops
