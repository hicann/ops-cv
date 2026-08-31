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
 * \file col2_im_v2_infershape.cpp
 * \brief Col2ImV2 infershape v2.0（与 canndev col2im_v2.cc 语义一致）
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_OUTPUT_SIZE = 1;
static constexpr size_t INPUT_IDX_KERNEL_SIZE = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t X_RANK = 3;
static constexpr size_t SIZE_TENSOR_LEN = 2;
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_C = 1;
static constexpr int64_t UNKNOWN_DIM = -1; // unknown shape 维度占位（dim=-1 传递）

// 读取长度 2 的 const tensor 值（值依赖输入）
static ge::graphStatus GetSizeValue(gert::InferShapeContext* context, size_t idx, const char* name, int64_t& sizeH,
                                    int64_t& sizeW)
{
    gert::Shape valueShape;
    OP_CHECK_IF(
        !Ops::Base::GetConstIntToShape(context, static_cast<int64_t>(idx), valueShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), name, "non-const tensor", "must be const tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueShape.GetDimNum() != SIZE_TENSOR_LEN,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), name,
                                              std::to_string(valueShape.GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    sizeH = valueShape[0];
    sizeW = valueShape[1];
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Col2ImV2(gert::InferShapeContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto outputSizeShape = context->GetInputShape(INPUT_IDX_OUTPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSizeShape);
    auto kernelSizeShape = context->GetInputShape(INPUT_IDX_KERNEL_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSizeShape);
    auto yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 1. 输入 dtype 校验（与 tiling ValidateDtype 对齐；GetInputDesc 写法参照 crop_infershape）
    const auto* xDesc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "FLOAT/FLOAT16"),
        return ge::GRAPH_FAILED);
    for (size_t idx : {INPUT_IDX_OUTPUT_SIZE, INPUT_IDX_KERNEL_SIZE}) {
        const auto* desc = context->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(),
                                              (idx == INPUT_IDX_OUTPUT_SIZE ? "output_size" : "kernel_size"),
                                              Ops::Base::ToString(desc->GetDataType()).c_str(), "INT32"),
                    return ge::GRAPH_FAILED);
    }

    // 2. 任一输入 UnknownRank → 输出 UnknownRank
    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*outputSizeShape) ||
        Ops::Base::IsUnknownRank(*kernelSizeShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    // 3. 校验 x 为 3-D
    OP_CHECK_IF(xShape->GetDimNum() != X_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x",
                                             (std::to_string(xShape->GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);

    // 4. 读取 const tensor 值
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t kernelH = 0;
    int64_t kernelW = 0;
    OP_CHECK_IF(GetSizeValue(context, INPUT_IDX_OUTPUT_SIZE, "output_size", outH, outW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Get output_size value failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSizeValue(context, INPUT_IDX_KERNEL_SIZE, "kernel_size", kernelH, kernelW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Get kernel_size value failed"), return ge::GRAPH_FAILED);

    // 5. unknown shape（dim=-1）短路：c 维依赖 dim(1)/(kH*kW)，unknown 传递
    // （outH/outW 来自 const 值总是具体；dim(0) 直接传递，unknown 与否均正确）
    if (Ops::Base::IsUnknownShape(*xShape)) {
        yShape->SetDimNum(0);
        yShape->AppendDim(xShape->GetDim(DIM_N));
        yShape->AppendDim(UNKNOWN_DIM);
        yShape->AppendDim(outH);
        yShape->AppendDim(outW);
        return ge::GRAPH_SUCCESS;
    }

    // 6. 校验 kH*kW != 0（除零保护）
    int64_t kernelArea = kernelH * kernelW;
    OP_CHECK_IF(kernelArea == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "kernel_size",
                    ("(" + std::to_string(kernelH) + ", " + std::to_string(kernelW) + ")").c_str(),
                    "kernel size product must not be 0"),
                return ge::GRAPH_FAILED);

    // 7. y = (x[0], x[1]/(kH*kW), outH, outW)
    yShape->SetDimNum(0);
    yShape->AppendDim(xShape->GetDim(DIM_N));
    yShape->AppendDim(xShape->GetDim(DIM_C) / kernelArea);
    yShape->AppendDim(outH);
    yShape->AppendDim(outW);
    OP_LOGI(context->GetNodeName(), "Col2ImV2 output shape: %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Col2ImV2)
    .InferShape(InferShape4Col2ImV2)
    .InputsDataDependency({INPUT_IDX_OUTPUT_SIZE, INPUT_IDX_KERNEL_SIZE});
} // namespace ops
