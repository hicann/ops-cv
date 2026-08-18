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
 * \file crop_and_resize_infershape.cpp
 * \brief InferShape implementation for crop_and_resize operator
 *
 * Output shape: [num_boxes, crop_height, crop_width, depth]
 *   num_boxes = boxes.shape[0]
 *   crop_height = crop_size[0] (value dependency)
 *   crop_width = crop_size[1] (value dependency)
 *   depth = x.shape[3]
 * Output dtype = boxes dtype (handled by def.cpp DataType config)
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "crop_and_resize_constraints.h"

#include <string>

using namespace ge;

namespace ops {

// 约束阈值常量和维度/索引常量来自 crop_and_resize_constraints.h

static ge::graphStatus InferShapeCropAndResize(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCropAndResize");

    // 获取输入 shape
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* boxesShape = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    const gert::Shape* cropSizeShape = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShape);

    // unknown rank 传播: x 或 boxes 为 -2 时，输出设为 unknown rank
    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*boxesShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank");
        gert::Shape* yShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    // 约束1: x 必须为 4D
    if (xShape->GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape->GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }

    // 约束11: boxes.shape[1] == 4
    if (boxesShape->GetDimNum() != BOXES_DIM || boxesShape->GetDim(1) != BOX_COORDS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape->GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }

    // 约束12: crop_size.shape == (2,)
    if (cropSizeShape->GetDimNum() != CROP_SIZE_DIM || cropSizeShape->GetDim(0) != CROP_SIZE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "crop_size",
                                      std::to_string(cropSizeShape->GetDim(0)).c_str(),
                                      std::to_string(CROP_SIZE_LEN).c_str());
        return ge::GRAPH_FAILED;
    }

    // 读取 crop_size 值（值依赖，input index = IDX_CROP_SIZE）
    const gert::Tensor* cropSizeTensor = context->GetInputTensor(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeTensor);

    int64_t cropHeight = ge::UNKNOWN_DIM;
    int64_t cropWidth = ge::UNKNOWN_DIM;

    // crop_size 非常量时（动态 shape），编译期无法获取具体值
    // 设输出维度为 UNKNOWN_DIM，跳过约束检查，由 tiling 阶段兜底
    if (cropSizeTensor->GetAddr() != nullptr) {
        const int32_t* cropSizeData = cropSizeTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeData);
        cropHeight = static_cast<int64_t>(cropSizeData[0]);
        cropWidth = static_cast<int64_t>(cropSizeData[1]);

        // 约束4 前置: crop_height/crop_width 必须 > 0
        if (cropHeight <= 0 || cropWidth <= 0) {
            std::string valMsg = "[" + std::to_string(cropHeight) + ", " + std::to_string(cropWidth) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                                  "crop_height and crop_width must be positive");
            return ge::GRAPH_FAILED;
        }

        // 约束4: max(crop_h, crop_w) <= 16
        if (cropHeight > CROP_DIM_MAX || cropWidth > CROP_DIM_MAX) {
            std::string valMsg = "[" + std::to_string(cropHeight) + ", " + std::to_string(cropWidth) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                                  "max(crop_h, crop_w) must be <= " + std::to_string(CROP_DIM_MAX));
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(context->GetNodeName(), "crop_size is non-const tensor, set output dims to UNKNOWN_DIM");
    }

    // 设置输出 shape: [num_boxes, crop_height, crop_width, depth]
    // unknown dim (-1) 通过 GetDim 直接传递到输出，无需特殊处理
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(X_DIM);
    yShape->SetDim(0, boxesShape->GetDim(0)); // num_boxes（-1 时自然传递）
    yShape->SetDim(1, cropHeight);            // crop_height
    yShape->SetDim(2, cropWidth);             // crop_width
    yShape->SetDim(3, xShape->GetDim(3));     // depth（-1 时自然传递）

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCropAndResize");
    return GRAPH_SUCCESS;
}

// 注册 infershape + 值依赖（crop_size index=3）
IMPL_OP_INFERSHAPE(CropAndResize).InferShape(InferShapeCropAndResize).InputsDataDependency({IDX_CROP_SIZE});
} // namespace ops
