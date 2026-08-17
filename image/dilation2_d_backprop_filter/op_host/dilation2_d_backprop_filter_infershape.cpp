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
 * \file dilation2_d_backprop_filter_infershape.cpp
 * \brief InferShape implementation for dilation2_d_backprop_filter operator
 *
 * Output shape = filter shape (SE §5.5)
 * Output dtype = x dtype (SE §5.6, framework auto-derive)
 * Supports both NHWC and NCHW data formats (v2.5)
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "op_common/op_host/util/shape_util.h"
#include <string>

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t RANK_4D = 4;
static constexpr int64_t RANK_3D = 3;
static constexpr int64_t UNKNOWN_DIM = -1;

static inline bool BothKnownAndNotEqual(int64_t a, int64_t b) { return a != UNKNOWN_DIM && b != UNKNOWN_DIM && a != b; }

static ge::graphStatus InferShapeDilation2DBackpropFilter(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDilation2DBackpropFilter");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* filterShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* outBpShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outBpShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Unknown rank(-2): if any input is unknownrank, output is unknownrank
    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*filterShape) ||
        Ops::Base::IsUnknownRank(*outBpShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank");
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    // Validate ranks: x=4D, filter=3D, out_backprop=4D
    OP_CHECK_IF(
        xShape->GetDimNum() != static_cast<size_t>(RANK_4D),
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(), "4"),
        return GRAPH_FAILED);
    OP_CHECK_IF(filterShape->GetDimNum() != static_cast<size_t>(RANK_3D),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter",
                                             std::to_string(filterShape->GetDimNum()).c_str(), "3"),
                return GRAPH_FAILED);
    OP_CHECK_IF(outBpShape->GetDimNum() != static_cast<size_t>(RANK_4D),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "out_backprop",
                                             std::to_string(outBpShape->GetDimNum()).c_str(), "4"),
                return GRAPH_FAILED);

    // Validate dtype: only DT_FLOAT is supported, all inputs must have the same dtype
    auto xDesc = context->GetInputDesc(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto filterDesc = context->GetInputDesc(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    auto outBpDesc = context->GetInputDesc(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outBpDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType filterDtype = filterDesc->GetDataType();
    ge::DataType outBpDtype = outBpDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "DT_FLOAT"),
                return GRAPH_FAILED);
    OP_CHECK_IF(filterDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "filter", Ops::Base::ToString(filterDtype).c_str(),
                                          "DT_FLOAT"),
                return GRAPH_FAILED);
    OP_CHECK_IF(outBpDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "out_backprop",
                                          Ops::Base::ToString(outBpDtype).c_str(), "DT_FLOAT"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        xDtype != filterDtype || xDtype != outBpDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, filter, out_backprop",
                                               (Ops::Base::ToString(xDtype) + ", " + Ops::Base::ToString(filterDtype) +
                                                ", " + Ops::Base::ToString(outBpDtype))
                                                   .c_str(),
                                               "all inputs must have the same dtype"),
        return GRAPH_FAILED);

    // Validate data_format: "NHWC" or "NCHW" (v2.5: NCHW support)
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* dataFormatPtr = attrs->GetStr(5);
    OP_CHECK_IF(
        dataFormatPtr == nullptr || (std::string(dataFormatPtr) != "NHWC" && std::string(dataFormatPtr) != "NCHW"),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format",
                                  dataFormatPtr != nullptr ? dataFormatPtr : "null", "NHWC or NCHW"),
        return GRAPH_FAILED);
    bool isNCHW = (dataFormatPtr != nullptr && std::string(dataFormatPtr) == "NCHW");

    // Validate strides N/C dims must be 1
    // NHWC: strides[0]==1, strides[3]==1; NCHW: strides[0]==1, strides[1]==1
    const auto* stridesVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesVec);
    OP_CHECK_IF(stridesVec->GetSize() < 4,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                          std::to_string(stridesVec->GetSize()).c_str(), "4 elements"),
                return GRAPH_FAILED);
    const int64_t* stridesData = stridesVec->GetData();
    if (isNCHW) {
        OP_CHECK_IF(stridesData[0] != 1 || stridesData[1] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", "strides[0] or strides[1] != 1", "1"),
                    return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(stridesData[0] != 1 || stridesData[3] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", "strides[0] or strides[3] != 1", "1"),
                    return GRAPH_FAILED);
    }

    // Validate rates N/C dims must be 1
    // NHWC: rates[0]==1, rates[3]==1; NCHW: rates[0]==1, rates[1]==1
    const auto* ratesVec = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ratesVec);
    OP_CHECK_IF(ratesVec->GetSize() < 4,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", std::to_string(ratesVec->GetSize()).c_str(),
                                          "4 elements"),
                return GRAPH_FAILED);
    const int64_t* ratesData = ratesVec->GetData();
    if (isNCHW) {
        OP_CHECK_IF(ratesData[0] != 1 || ratesData[1] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", "rates[0] or rates[1] != 1", "1"),
                    return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(ratesData[0] != 1 || ratesData[3] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", "rates[0] or rates[3] != 1", "1"),
                    return GRAPH_FAILED);
    }

    // Validate depth consistency based on data_format
    // NHWC: x.C(dim3) == filter.C(dim2) == out_bp.C(dim3)
    // NCHW: x.C(dim1) == filter.C(dim0) == out_bp.C(dim1)
    if (isNCHW) {
        OP_CHECK_IF(BothKnownAndNotEqual(xShape->GetDim(1), filterShape->GetDim(0)) ||
                        BothKnownAndNotEqual(xShape->GetDim(1), outBpShape->GetDim(1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop", "x.C, filter.C, out_bp.C",
                        "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
                    return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(BothKnownAndNotEqual(xShape->GetDim(3), filterShape->GetDim(2)) ||
                        BothKnownAndNotEqual(xShape->GetDim(3), outBpShape->GetDim(3)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop", "x.C, filter.C, out_bp.C",
                        "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
                    return GRAPH_FAILED);
    }

    // Output shape = filter shape (SE §5.5)
    yShape->SetDimNum(filterShape->GetDimNum());
    for (size_t i = 0; i < filterShape->GetDimNum(); i++) {
        yShape->SetDim(i, filterShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeDilation2DBackpropFilter");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Dilation2DBackpropFilter).InferShape(InferShapeDilation2DBackpropFilter);
} // namespace ops
