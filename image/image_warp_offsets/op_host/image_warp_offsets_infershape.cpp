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
 * \file image_warp_offsets_infershape.cpp
 * \brief
 */

// ---------------IMGWarpOffsets Op start-------------------

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kImagesIndex = 0U;
constexpr size_t kOffsetsIndex = 1U;
constexpr size_t kWarpImagesIndex = 0U;
constexpr size_t kInputRank = 4U;
constexpr size_t kOutputRank = 5U;
constexpr size_t kBatchDim = 0U;
constexpr size_t kOffsetsPointDim = 1U;
constexpr size_t kOffsetsHeightDim = 2U;
constexpr size_t kOffsetsWidthDim = 3U;
constexpr size_t kImagesChannelDim = 3U;
constexpr int64_t kPointNum = 4;
constexpr int64_t kChannelNum = 3;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForIMGWarpOffsets(gert::InferShapeContext* context)
{
    const gert::Shape* imagesShape = context->GetInputShape(kImagesIndex);
    const gert::Shape* offsetsShape = context->GetInputShape(kOffsetsIndex);
    gert::Shape* warpImagesShape = context->GetOutputShape(kWarpImagesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, imagesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, warpImagesShape);

    if (Ops::Base::IsUnknownRank(*imagesShape) || Ops::Base::IsUnknownRank(*offsetsShape)) {
        Ops::Base::SetUnknownRank(*warpImagesShape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(imagesShape->GetDimNum() != kInputRank,
                OP_LOGE(context, "images must be rank 4, but got %zu", imagesShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        imagesShape->GetDim(kImagesChannelDim) != kChannelNum &&
            imagesShape->GetDim(kImagesChannelDim) != ge::UNKNOWN_DIM,
        OP_LOGE(context, "images last dimension must be 3, but got %ld", imagesShape->GetDim(kImagesChannelDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(offsetsShape->GetDimNum() != kInputRank,
                OP_LOGE(context, "offsets must be rank 4, but got %zu", offsetsShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        offsetsShape->GetDim(kOffsetsPointDim) != kPointNum &&
            offsetsShape->GetDim(kOffsetsPointDim) != ge::UNKNOWN_DIM,
        OP_LOGE(context, "offsets second dimension must be 4, but got %ld", offsetsShape->GetDim(kOffsetsPointDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(imagesShape->GetDim(kBatchDim) != offsetsShape->GetDim(kBatchDim),
                OP_LOGE(context, "images and offsets batch dimensions must be equal, but got %ld and %ld",
                        imagesShape->GetDim(kBatchDim), offsetsShape->GetDim(kBatchDim)),
                return ge::GRAPH_FAILED);

    warpImagesShape->SetDimNum(kOutputRank);
    warpImagesShape->SetDim(kBatchDim, offsetsShape->GetDim(kBatchDim));
    warpImagesShape->SetDim(kOffsetsPointDim, offsetsShape->GetDim(kOffsetsPointDim));
    warpImagesShape->SetDim(kOffsetsHeightDim, offsetsShape->GetDim(kOffsetsHeightDim));
    warpImagesShape->SetDim(kOffsetsWidthDim, offsetsShape->GetDim(kOffsetsWidthDim));
    warpImagesShape->SetDim(kImagesChannelDim + 1U, imagesShape->GetDim(kImagesChannelDim));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IMGWarpOffsets).InferShape(InferShapeForIMGWarpOffsets);
} // namespace ops
