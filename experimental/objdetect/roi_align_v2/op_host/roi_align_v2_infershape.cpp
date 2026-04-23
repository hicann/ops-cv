/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 # Authors (accounts):
 # - Liu Jun <@kbryantttt>
 # - Tu Yuanhang <@TuYHAAAAAA>
 # - Zhou Jianhua<@LePenseur>
 # - Liang Yanglin <@liang-yanglin>
 # - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file roi_align_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRoiAlignV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRoiAlignV2");

    const gert::Shape *features_shape = context->GetInputShape(0);
    const gert::Shape *rois_shape = context->GetInputShape(1);
    gert::Shape *output_shape = context->GetOutputShape(0);

    uint32_t numRois = rois_shape->GetDim(0);
    uint32_t channels = features_shape->GetDim(1);

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