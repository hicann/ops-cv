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
 * \file resize_bilinear_v2_infershape.cpp
 * \brief resize_bilinear_v2_infershape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;
namespace ops {

struct OutInfo {
    int64_t output_h{0};
    int64_t output_w{0};
    int64_t output_d{0};
};

static constexpr size_t IN_X = 0;
static constexpr size_t IN_SIZE = 1;
static constexpr size_t ATTR_2_IDX = 2;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SIZE_NUM_2D = 2;
static const int64_t OUTPUT_DIM_NUM = 4;

template <typename T>
static bool GetSizeValueFor2D(
    const gert::InferShapeContext* context, const gert::Tensor* size_tensor, OutInfo& out_size)
{
    const T* size_value = size_tensor->GetData<T>();
    const size_t size_num = size_tensor->GetShapeSize();
    OP_CHECK_IF(
        size_num != SIZE_NUM_2D, OP_LOGE(context->GetNodeName(), "The size number must be 2, but is %d", size_num),
        return false);

    out_size.output_h = static_cast<int64_t>(size_value[0]);
    out_size.output_w = static_cast<int64_t>(size_value[1]);

    return true;
}

static bool GetSizeFor2D(const gert::InferShapeContext* context, const gert::Tensor* size_tensor, OutInfo& out_size)
{
    if (!Ops::Cv::IsConstTensor(size_tensor)) {
        OP_LOGW(context->GetNodeName(), "the input is not const tensor, out_size will be -1.");
        out_size.output_h = ge::UNKNOWN_DIM;
        out_size.output_w = ge::UNKNOWN_DIM;
        return true;
    }
    ge::DataType size_dtype = size_tensor->GetDataType();
    switch (size_dtype) {
        case ge::DT_INT32: {
            OP_CHECK_IF(
                !GetSizeValueFor2D<int32_t>(context, size_tensor, out_size),
                OP_LOGE(context->GetNodeName(), "Get size const for int32 failed!"), return false);
            break;
        }
        case ge::DT_INT64: {
            OP_CHECK_IF(
                !GetSizeValueFor2D<int64_t>(context, size_tensor, out_size),
                OP_LOGE(context->GetNodeName(), "Get size const for int64 failed!"), return false);
            break;
        }
        default:
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(
                context->GetNodeName(), "size", Ops::Base::ToString(size_dtype).c_str(), "[int32, int64]");
            return false;
    }
    return true;
}

static bool ResizeInfershapeFor2D(
    const gert::InferShapeContext* context, const gert::Shape* x_shape, const ge::Format input_format,
    const OutInfo& out_info, gert::Shape* y_shape)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ResizeInfershape");
    OP_LOGD(context->GetNodeName(), "input x shape = %s", Ops::Base::ToString(*x_shape).c_str());
    OP_LOGD(context->GetNodeName(), "input x format = %s", Ops::Base::ToString(input_format).c_str());
    OP_CHECK_IF(
        input_format != FORMAT_NHWC && input_format != FORMAT_NCHW,
        OP_LOGE(
            context->GetNodeName(), "input format only support [NHWC,NCHW], but is %s",
            Ops::Base::ToString(input_format).c_str()),
        return false);

    // -2 infer shape
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        OP_LOGD(context->GetNodeName(), "x_shape is UnknownRank, set output_shape to (-1, -1, -1, -1)");
        Ops::Base::SetUnknownShape(OUTPUT_DIM_NUM, *y_shape);
    } else {
        constexpr size_t output_len = OUTPUT_DIM_NUM;
        const size_t input_dim_size = x_shape->GetDimNum();
        OP_CHECK_IF(
            input_dim_size != output_len,
            OP_LOGE(
                context->GetNodeName(), "input shape only support 4D, but is %s",
                Ops::Base::ToString(*x_shape).c_str()),
            return false);

        *y_shape = *x_shape;
    }

    const size_t image_h_idx = input_format == FORMAT_NHWC ? 1 : 2;
    const size_t image_w_idx = input_format == FORMAT_NHWC ? 2 : 3;
    y_shape->SetDim(image_h_idx, out_info.output_h);
    y_shape->SetDim(image_w_idx, out_info.output_w);

    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do ResizeInfershape");

    return true;
}

ge::graphStatus InferShape4Resize2DWithConstSize(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do InferShape4Resize2DWithConstSize");
    const gert::Shape* x_shape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::Tensor* size_tensor = context->GetInputTensor(IN_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, size_tensor);

    OutInfo out_size;
    OP_LOGD(context->GetNodeName(), "begin to get size from input %ld.", IN_SIZE);
    OP_CHECK_IF(
        !GetSizeFor2D(context, size_tensor, out_size), OP_LOGE(context->GetNodeName(), "Get size const failed!"),
        return ge::GRAPH_FAILED);

    auto x_rutime_desc = context->GetInputDesc(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_rutime_desc);
    ge::Format input_format = x_rutime_desc->GetOriginFormat();
    OP_CHECK_IF(
        !ResizeInfershapeFor2D(context, x_shape, input_format, out_size, y_shape),
        OP_LOGE(context->GetNodeName(), "Do ResizeInfershape failed!"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Do InferShape4Resize2DWithConstSize success");

    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4ResizeBilinearV2(gert::InferDataTypeContext* context)
{
    OP_LOGD("ResizeBilinearV2", "InferDtype4ResizeBilinearV2 enter");
    if (context == nullptr) {
        OP_LOGE("ResizeBilinearV2", "InferDtype4ResizeBilinearV2 context is null.");
        return GRAPH_FAILED;
    }

    OP_LOGI(context->GetNodeName(), "set output to default dtype float");
    context->SetOutputDataType(0, ge::DT_FLOAT);
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        OP_LOGI(context->GetNodeName(), "attrsPtr is nullptr");
        return GRAPH_SUCCESS;
    }

    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(ATTR_2_IDX);
    if (dstDtype == nullptr) {
        OP_LOGI(context->GetNodeName(), "dstDtype is not configed");
        return GRAPH_SUCCESS;
    }

    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_LOGD(context->GetNodeName(), "dtype in attrs is %ld", (int64_t)outDtype);
    OP_CHECK_IF(
        !(outDtype == ge::DT_FLOAT || outDtype == ge::DT_FLOAT16 || outDtype == ge::DT_BF16 ||
          outDtype == ge::DT_UINT8),
        OP_LOGE(context->GetNodeName(), "dtype should be float32, float16, bfloat16 or uint8."), return GRAPH_FAILED);
    auto xDtype = context->GetInputDataType(0);
    OP_LOGD(context->GetNodeName(), "xDtype is %ld", (int64_t)xDtype);

    OP_CHECK_IF(
        (xDtype == ge::DT_FLOAT && (outDtype == ge::DT_FLOAT16 || outDtype == ge::DT_BF16)),
        OP_LOGE(context->GetNodeName(), "xDtype is float32, outDtype should not be float16 or bfloat16."),
        return GRAPH_FAILED);
    OP_CHECK_IF(
        ((xDtype == ge::DT_FLOAT16 && outDtype == ge::DT_BF16) ||
         (xDtype == ge::DT_BF16 && outDtype == ge::DT_FLOAT16)),
        OP_LOGE(context->GetNodeName(), "xDtype is float16 or bfloat16, outDtype should be same to xDtype or float32."),
        return GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "set output to configed dtype: %s", Ops::Base::ToString(outDtype).c_str());
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "Do InferDtype4ResizeBilinearV2 success");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeBilinearV2)
    .InferShape(InferShape4Resize2DWithConstSize)
    .InputsDataDependency({IN_SIZE})
    .InferDataType(InferDtype4ResizeBilinearV2);
} // namespace ops
