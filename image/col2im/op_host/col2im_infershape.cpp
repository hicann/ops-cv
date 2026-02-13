/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file col2im_infershape.cpp
 * \brief col2im_infershape
 */

#include <numeric>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;
using namespace std;

namespace ops {
constexpr int64_t X_IDX = 0;
constexpr int64_t OUTPUT_SIZE_IDX = 1;
constexpr int64_t Y_IDX = 0;
constexpr int64_t N_DIM = 0;
constexpr int64_t C_DIM = 1;
constexpr int64_t H_DIM = 2;
constexpr int64_t W_DIM = 3;
struct OutputSizeInfo {
  int64_t output_size_h = 0;
  int64_t output_size_w = 0;
};

static bool GetOutputSizeValue(const gert::InferShapeContext *context, const gert::Tensor *output_size_tensor,
                               OutputSizeInfo &output_size_info) {
  const int32_t *output_size_value = output_size_tensor->GetData<int32_t>();
  const size_t output_size_num = static_cast<size_t>(output_size_tensor->GetShapeSize());
  OP_CHECK_IF(output_size_num != 2,
           OP_LOGE(
               context->GetNodeName(), "The length of output_size must be 2, but got %u.", output_size_num),
           return false);
  OP_LOGD(context->GetNodeName(), "get output size length %u", output_size_num);
  output_size_info.output_size_h = static_cast<int64_t>(output_size_value[0]);
  output_size_info.output_size_w = static_cast<int64_t>(output_size_value[1]);

  return true;
}

static bool GetOutputSize(const gert::InferShapeContext *context, const gert::Tensor *output_size_tensor,
                          OutputSizeInfo &output_size_info) {
    if (!Ops::Cv::IsConstTensor(output_size_tensor)) {
        OP_LOGE(context->GetNodeName(), "the input [output_size] is not const tensor.");
        return false;
    }
    OP_CHECK_IF(!GetOutputSizeValue(context, output_size_tensor, output_size_info),
            OP_LOGE(context->GetNodeName(), "Get size const for int32 failed!"),
            return false);
    return true;
}

static ge::graphStatus InferShape4Col2im(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "begin to do InferShape4Col2im.");
    auto output_size_tensor = context->GetInputTensor(OUTPUT_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_tensor);

    auto x_shape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_LOGD(context->GetNodeName(), "input x shape = %s", Ops::Base::ToString(*x_shape).c_str());
    auto output_size_shape = context->GetInputShape(OUTPUT_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_shape);
    auto y_shape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    if (Ops::Base::IsUnknownRank(*x_shape) || Ops::Base::IsUnknownRank(*output_size_shape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank.");
        OP_LOGI(context->GetNodeName(), "Do InferShape4Col2im success.");
        Ops::Base::SetUnknownRank(*y_shape);
        return GRAPH_SUCCESS;
    }

    auto x_shape_size = x_shape->GetDimNum();
    OP_CHECK_IF((x_shape_size != 4),
            OP_LOGE(
                context->GetNodeName(), "x's dim length should be 4, but got %s.", Ops::Base::ToString(*x_shape).c_str()),
            return ge::GRAPH_FAILED);
    y_shape->SetDimNum(x_shape_size);

    OutputSizeInfo output_size_info;
    OP_LOGD(context->GetNodeName(), "begin to get output size from input [output_size].");
    OP_CHECK_IF(!GetOutputSize(context, output_size_tensor, output_size_info),
            OP_LOGE(context->GetNodeName(), "Get output_size const failed!"),
            return ge::GRAPH_FAILED);

    y_shape->SetDim(N_DIM, x_shape->GetDim(N_DIM));
    y_shape->SetDim(C_DIM, x_shape->GetDim(C_DIM));
    y_shape->SetDim(H_DIM, output_size_info.output_size_h);
    y_shape->SetDim(W_DIM, output_size_info.output_size_w);
    OP_LOGD(context->GetNodeName(), "output y shape = %s", Ops::Base::ToString(*y_shape).c_str());

    OP_LOGI(context->GetNodeName(), "Do InferShape4Col2im success.");
    return GRAPH_SUCCESS;
}


graphStatus InferDtype4Col2im(gert::InferDataTypeContext *context)
{
    const auto dataColDTypeInfer = context->GetInputDataType(X_IDX);
    context->SetOutputDataType(Y_IDX, dataColDTypeInfer);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Col2im).InferShape(InferShape4Col2im).InferDataType(InferDtype4Col2im);
} // namespace ops