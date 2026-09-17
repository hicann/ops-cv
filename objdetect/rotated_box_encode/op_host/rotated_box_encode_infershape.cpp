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
 * \file rotated_box_encode_infershape.cpp
 * \brief RotatedBoxEncode shape inference: y.shape == anchor_box.shape
 */

// op_impl_registry.h: provides IMPL_OP_INFERSHAPE macro, gert::InferShapeContext.
#include "register/op_impl_registry.h"

#include <cstdint>

#include "graph/types.h"

using namespace ge;

namespace ops {

// InferShapeForRotatedBoxEncode: GE shape inference callback.
//   Per proto.md §4: both inputs and the output are identical 3-D (B, 5, N)
//   tensors; broadcasting is not supported.
static ge::graphStatus InferShapeForRotatedBoxEncode(gert::InferShapeContext* context)
{
    const gert::Shape* anchorShape = context->GetInputShape(0);
    const gert::Shape* gtShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (anchorShape == nullptr || gtShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (anchorShape->GetDimNum() != 3 || gtShape->GetDimNum() != 3 || anchorShape->GetDim(1) != 5 ||
        gtShape->GetDim(1) != 5) {
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (anchorShape->GetDim(i) != gtShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    yShape->SetDimNum(3);
    for (size_t i = 0; i < 3; ++i) {
        yShape->SetDim(i, anchorShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

// [REF_SAMPLE] — original AddCustom sample InferShape logic (rank matching,
// UNKNOWN_DIM / UNKNOWN_RANK resolution).  Preserved verbatim for reference; NOT compiled.
//
// namespace {
// bool IsUnknownRank(const gert::Shape* shape)
// {
//     return shape != nullptr && shape->GetDimNum() == 1 &&
//            shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
// }
// } // namespace
//
// static ge::graphStatus InferShapeForRotatedBoxEncode(gert::InferShapeContext* context)
// {
//     const gert::Shape* xShape = context->GetInputShape(0);
//     const gert::Shape* yShape = context->GetInputShape(1);
//     gert::Shape* zShape = context->GetOutputShape(0);
//     if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
//         return GRAPH_FAILED;
//     }
//     if (IsUnknownRank(xShape) || IsUnknownRank(yShape)) {
//         zShape->SetDimNum(1);
//         zShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
//         return GRAPH_SUCCESS;
//     }
//     const size_t rank = xShape->GetDimNum();
//     if (rank != yShape->GetDimNum()) {
//         return GRAPH_FAILED;
//     }
//     std::vector<int64_t> outputDims(rank);
//     for (size_t i = 0; i < rank; ++i) {
//         const int64_t xDim = xShape->GetDim(i);
//         const int64_t yDim = yShape->GetDim(i);
//         if (xDim == yDim) {
//             outputDims[i] = xDim;
//         } else if (xDim == ge::UNKNOWN_DIM) {
//             outputDims[i] = yDim;
//         } else if (yDim == ge::UNKNOWN_DIM) {
//             outputDims[i] = xDim;
//         } else {
//             return GRAPH_FAILED;
//         }
//     }
//     zShape->SetDimNum(rank);
//     for (size_t i = 0; i < rank; ++i) {
//         zShape->SetDim(i, outputDims[i]);
//     }
//     return GRAPH_SUCCESS;
// }

IMPL_OP_INFERSHAPE(RotatedBoxEncode).InferShape(InferShapeForRotatedBoxEncode);

} // namespace ops
