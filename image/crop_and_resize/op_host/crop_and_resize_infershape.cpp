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

// 校验输入 shape：维度数、空 tensor、boxes.shape[1]
static ge::graphStatus ValidateInputShapes(gert::InferShapeContext* context, const gert::Shape* xShape,
                                           const gert::Shape* boxesShape, const gert::Shape* cropSizeShape)
{
    if (xShape->GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape->GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < xShape->GetDimNum(); i++) {
        if (xShape->GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x",
                                                  std::to_string(xShape->GetDim(i)).c_str(),
                                                  ("x.shape[" + std::to_string(i) + "] must not be zero").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (boxesShape->GetDimNum() != BOXES_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape->GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (boxesShape->GetDim(1) != ge::UNKNOWN_DIM && boxesShape->GetDim(1) != BOX_COORDS) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "boxes", std::to_string(boxesShape->GetDim(1)).c_str(), "boxes.shape[1] must be 4");
        return ge::GRAPH_FAILED;
    }
    if (boxesShape->GetDim(0) == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes",
                                              std::to_string(boxesShape->GetDim(0)).c_str(),
                                              "boxes.shape[0] must not be zero");
        return ge::GRAPH_FAILED;
    }
    if (cropSizeShape->GetDimNum() != CROP_SIZE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "crop_size",
                                     (std::to_string(cropSizeShape->GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (cropSizeShape->GetDim(0) != CROP_SIZE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "crop_size",
                                                  std::to_string(cropSizeShape->GetDim(0)).c_str(),
                                                  "crop_size.shape[0] must be 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 读取 crop_size 值并校验 dtype 和范围
static ge::graphStatus ReadCropSizeValue(gert::InferShapeContext* context, int64_t& cropHeight, int64_t& cropWidth)
{
    const gert::Tensor* cropSizeTensor = context->GetInputTensor(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeTensor);

    ge::DataType cropSizeDtype = context->GetInputDesc(IDX_CROP_SIZE)->GetDataType();
    if (cropSizeDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "crop_size", Ops::Base::ToString(cropSizeDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    cropHeight = ge::UNKNOWN_DIM;
    cropWidth = ge::UNKNOWN_DIM;

    if (cropSizeTensor->GetAddr() != nullptr) {
        const int32_t* cropSizeData = cropSizeTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeData);
        cropHeight = static_cast<int64_t>(cropSizeData[0]);
        cropWidth = static_cast<int64_t>(cropSizeData[1]);

        if (cropHeight <= 0 || cropWidth <= 0) {
            std::string valMsg = "[" + std::to_string(cropHeight) + ", " + std::to_string(cropWidth) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                                  "crop_height and crop_width must be positive");
            return ge::GRAPH_FAILED;
        }
        // 注：crop<=16 上限不在此检查。该上限是 AiCore tiling 约束而非算子语义，
        // 由 def.cpp 的 CheckIfAICoreSupported 在引擎分配阶段拒绝并 fallback 到 AiCpu；
        // 此处拦截会导致 AiCpu 路径（可正确计算 crop>16）也无法通过。
    } else {
        OP_LOGD(context->GetNodeName(), "crop_size is non-const tensor, set output dims to UNKNOWN_DIM");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCropAndResize(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCropAndResize");

    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* boxesShape = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    const gert::Shape* cropSizeShape = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShape);

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*boxesShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank");
        gert::Shape* yShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF(ValidateInputShapes(context, xShape, boxesShape, cropSizeShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputShapes failed"), return ge::GRAPH_FAILED);

    int64_t cropHeight = ge::UNKNOWN_DIM;
    int64_t cropWidth = ge::UNKNOWN_DIM;
    OP_CHECK_IF(ReadCropSizeValue(context, cropHeight, cropWidth) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadCropSizeValue failed"), return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(X_DIM);
    yShape->SetDim(0, boxesShape->GetDim(0));
    yShape->SetDim(1, cropHeight);
    yShape->SetDim(2, cropWidth);
    yShape->SetDim(3, xShape->GetDim(3));

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCropAndResize");
    return GRAPH_SUCCESS;
}

// 注册 infershape + 值依赖（crop_size index=3）
IMPL_OP_INFERSHAPE(CropAndResize).InferShape(InferShapeCropAndResize).InputsDataDependency({IDX_CROP_SIZE});
} // namespace ops
