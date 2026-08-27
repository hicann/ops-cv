/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * This file incorporates source contributed by the OpenBOAT project at Harbin Institute of Technology (HIT).
 * Original contributors:
 * - Liu Jun <@kbryantttt>
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Zhou Jianhua <@LePenseur>
 * - Liang Yanglin <@liang-yanglin>
 * - Su Tonghua <@sutonghua>
 */

/*!
 * \file roi_align_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRoiAlignV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRoiAlignV2");

    const gert::Shape* features_shape = context->GetInputShape(0);
    const gert::Shape* rois_shape = context->GetInputShape(1);
    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, features_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rois_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    if (Ops::Base::IsUnknownRank(*features_shape) || Ops::Base::IsUnknownRank(*rois_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        return GRAPH_SUCCESS;
    }
    OP_CHECK_IF(features_shape->GetDimNum() != 4U,
                OP_LOGE(context, "features dim num must be 4, but got %zu.", features_shape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(rois_shape->GetDimNum() != 2U,
                OP_LOGE(context, "rois dim num must be 2, but got %zu.", rois_shape->GetDimNum()), return GRAPH_FAILED);
    OP_CHECK_IF(rois_shape->GetDim(1) != ge::UNKNOWN_DIM && rois_shape->GetDim(1) != 5,
                OP_LOGE(context, "rois second dimension must be 5, but got %ld.", rois_shape->GetDim(1)),
                return GRAPH_FAILED);

    int64_t numRois = rois_shape->GetDim(0);
    int64_t channels = features_shape->GetDim(1);

    int32_t pooledHeight = 0;
    int32_t pooledWidth = 0;

    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* heighthAttr = attrs->GetInt(0);
        if (heighthAttr != nullptr) {
            pooledHeight = static_cast<int32_t>(*heighthAttr);
        }
        const int64_t* widthAttr = attrs->GetInt(1);
        if (widthAttr != nullptr) {
            pooledWidth = static_cast<int32_t>(*widthAttr);
        }
    }
    output_shape->SetDimNum(4);
    output_shape->SetDim(0, numRois);
    output_shape->SetDim(1, channels);
    output_shape->SetDim(2, pooledHeight);
    output_shape->SetDim(3, pooledWidth);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeRoiAlignV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiAlignV2).InferShape(InferShapeRoiAlignV2);
} // namespace ops
