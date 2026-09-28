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
#include <algorithm>
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

// Compute theoretical forward output spatial dims, consistent with arch35 tiling ComputeOutputDims.
// out_backprop is the gradient of the forward output, so its H/W must match these dims.
static void ComputeForwardOutDims(int64_t strideH, int64_t strideW, int64_t rateH, int64_t rateW,
                                  const std::string& paddingMode, const int64_t* pads, bool ceilMode, int64_t inputH,
                                  int64_t inputW, int64_t filterH, int64_t filterW, int64_t& outH, int64_t& outW)
{
    int64_t windowH = (filterH - 1) * rateH + 1;
    int64_t windowW = (filterW - 1) * rateW + 1;

    if (paddingMode == "SAME") {
        outH = (inputH + strideH - 1) / strideH;
        outW = (inputW + strideW - 1) / strideW;
    } else if (paddingMode == "CALCULATED") {
        // pads: [top, bottom, left, right]
        if (ceilMode) {
            outH = (inputH - windowH + pads[0] + pads[1] + strideH - 1) / strideH + 1;
            outW = (inputW - windowW + pads[2] + pads[3] + strideW - 1) / strideW + 1;
        } else {
            outH = (inputH - windowH + pads[0] + pads[1]) / strideH + 1;
            outW = (inputW - windowW + pads[2] + pads[3]) / strideW + 1;
        }
    } else { // VALID
        outH = (inputH - windowH) / strideH + 1;
        outW = (inputW - windowW) / strideW + 1;
    }

    // Clamp to non-negative (same as arch35 tiling)
    outH = std::max(outH, static_cast<int64_t>(0));
    outW = std::max(outW, static_cast<int64_t>(0));
}

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
    const char* paddingModePtr = attrs->GetStr(2);
    OP_CHECK_IF(
        paddingModePtr == nullptr || (std::string(paddingModePtr) != "SAME" && std::string(paddingModePtr) != "VALID" &&
                                      std::string(paddingModePtr) != "CALCULATED"),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding_mode",
                                  paddingModePtr != nullptr ? paddingModePtr : "null", "SAME, VALID or CALCULATED"),
        return GRAPH_FAILED);
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
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                              ("strides[0]=" + std::to_string(stridesData[0]) +
                                               ", strides[1]=" + std::to_string(stridesData[1])),
                                              "1"),
                    return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(stridesData[0] != 1 || stridesData[3] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                              ("strides[0]=" + std::to_string(stridesData[0]) +
                                               ", strides[3]=" + std::to_string(stridesData[3])),
                                              "1"),
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
        OP_CHECK_IF(
            ratesData[0] != 1 || ratesData[1] != 1,
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "rates",
                ("rates[0]=" + std::to_string(ratesData[0]) + ", rates[1]=" + std::to_string(ratesData[1])), "1"),
            return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            ratesData[0] != 1 || ratesData[3] != 1,
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "rates",
                ("rates[0]=" + std::to_string(ratesData[0]) + ", rates[3]=" + std::to_string(ratesData[3])), "1"),
            return GRAPH_FAILED);
    }

    // Validate depth consistency based on data_format
    // NHWC: x.C(dim3) == filter.C(dim2) == out_bp.C(dim3)
    // NCHW: x.C(dim1) == filter.C(dim0) == out_bp.C(dim1)
    if (isNCHW) {
        OP_CHECK_IF(
            BothKnownAndNotEqual(xShape->GetDim(1), filterShape->GetDim(0)) ||
                BothKnownAndNotEqual(xShape->GetDim(1), outBpShape->GetDim(1)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, filter, out_backprop",
                ("x.C=" + std::to_string(xShape->GetDim(1)) + ", filter.C=" + std::to_string(filterShape->GetDim(0)) +
                 ", out_bp.C=" + std::to_string(outBpShape->GetDim(1))),
                "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
            return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            BothKnownAndNotEqual(xShape->GetDim(3), filterShape->GetDim(2)) ||
                BothKnownAndNotEqual(xShape->GetDim(3), outBpShape->GetDim(3)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, filter, out_backprop",
                ("x.C=" + std::to_string(xShape->GetDim(3)) + ", filter.C=" + std::to_string(filterShape->GetDim(2)) +
                 ", out_bp.C=" + std::to_string(outBpShape->GetDim(3))),
                "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
            return GRAPH_FAILED);
    }

    // Extract spatial strides/rates and validate ranges
    // NHWC: strides/rates=[1,sH,sW,1]; NCHW: strides/rates=[1,1,sH,sW]
    int64_t strideH = 0, strideW = 0, rateH = 0, rateW = 0;
    if (isNCHW) {
        strideH = stridesData[2];
        strideW = stridesData[3];
        rateH = ratesData[2];
        rateW = ratesData[3];
    } else {
        strideH = stridesData[1];
        strideW = stridesData[2];
        rateH = ratesData[1];
        rateW = ratesData[2];
    }
    OP_CHECK_IF(strideH < 1 || strideW < 1,
                OP_LOGE_FOR_INVALID_VALUE(
                    context->GetNodeName(), "strides",
                    ("strides spatial=[" + std::to_string(strideH) + ", " + std::to_string(strideW) + "]"), ">= 1"),
                return GRAPH_FAILED);
    OP_CHECK_IF(rateH < 1 || rateW < 1,
                OP_LOGE_FOR_INVALID_VALUE(
                    context->GetNodeName(), "rates",
                    ("rates spatial=[" + std::to_string(rateH) + ", " + std::to_string(rateW) + "]"), ">= 1"),
                return GRAPH_FAILED);

    // Validate out_backprop batch consistency: out_bp.N must be the same as x.N (both known)
    OP_CHECK_IF(BothKnownAndNotEqual(xShape->GetDim(0), outBpShape->GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, out_backprop",
                                                       ("x.N=" + std::to_string(xShape->GetDim(0)) +
                                                        ", out_bp.N=" + std::to_string(outBpShape->GetDim(0))),
                                                       "batch mismatch: out_backprop.N must be the same as x.N"),
                return GRAPH_FAILED);

    // Validate out_backprop spatial dims against theoretical forward output geometry.
    // out_backprop is the gradient of the forward output: its H/W must match the dims
    // derived from x/filter/strides/rates/padding. Only checked when the theoretical
    // dims are fully derivable (static x/filter spatial dims); dynamic dims (-1) are tolerated.
    int64_t inputH = 0, inputW = 0, filterH = 0, filterW = 0;
    if (isNCHW) {
        inputH = xShape->GetDim(2);
        inputW = xShape->GetDim(3);
        filterH = filterShape->GetDim(1);
        filterW = filterShape->GetDim(2);
    } else {
        inputH = xShape->GetDim(1);
        inputW = xShape->GetDim(2);
        filterH = filterShape->GetDim(0);
        filterW = filterShape->GetDim(1);
    }
    if (inputH != UNKNOWN_DIM && inputW != UNKNOWN_DIM && filterH != UNKNOWN_DIM && filterW != UNKNOWN_DIM) {
        // Read pads and ceil_mode attrs (needed by CALCULATED padding)
        const auto* padsVec = attrs->GetListInt(3);
        OP_CHECK_NULL_WITH_CONTEXT(context, padsVec);
        OP_CHECK_IF(padsVec->GetSize() < 4,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pads",
                                              std::to_string(padsVec->GetSize()).c_str(), "4 elements"),
                    return GRAPH_FAILED);
        const int64_t* padsData = padsVec->GetData();
        const bool* ceilModePtr = attrs->GetBool(4);
        OP_CHECK_NULL_WITH_CONTEXT(context, ceilModePtr);
        bool ceilMode = *ceilModePtr;

        int64_t expectOutH = 0, expectOutW = 0;
        ComputeForwardOutDims(strideH, strideW, rateH, rateW, std::string(paddingModePtr), padsData, ceilMode, inputH,
                              inputW, filterH, filterW, expectOutH, expectOutW);
        int64_t bpH = isNCHW ? outBpShape->GetDim(2) : outBpShape->GetDim(1);
        int64_t bpW = isNCHW ? outBpShape->GetDim(3) : outBpShape->GetDim(2);
        OP_CHECK_IF(BothKnownAndNotEqual(bpH, expectOutH) || BothKnownAndNotEqual(bpW, expectOutW),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop",
                        ("out_bp.H=" + std::to_string(bpH) + ", out_bp.W=" + std::to_string(bpW) +
                         ", expected H_out=" + std::to_string(expectOutH) + ", W_out=" + std::to_string(expectOutW)),
                        "spatial mismatch: out_backprop H/W must match forward output dims derived from "
                        "x/filter/strides/rates/padding"),
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
