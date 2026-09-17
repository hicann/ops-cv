/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "any_value.h"

namespace {

using Ops::Cv::AnyValue;

gert::InfershapeContextPara MakeBaseCase(const std::vector<int64_t>& strides, const std::vector<int64_t>& rates,
                                         const std::vector<int64_t>& pads, const std::string& paddingMode, int64_t outH,
                                         int64_t outW)
{
    return gert::InfershapeContextPara(
        "Dilation2DBackpropInput",
        {
            {{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, outH, outW, 1}, {1, outH, outW, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("strides", AnyValue::CreateFrom<std::vector<int64_t>>(strides)),
            gert::InfershapeContextPara::OpAttr("rates", AnyValue::CreateFrom<std::vector<int64_t>>(rates)),
            gert::InfershapeContextPara::OpAttr("padding_mode", AnyValue::CreateFrom<std::string>(paddingMode)),
            gert::InfershapeContextPara::OpAttr("pads", AnyValue::CreateFrom<std::vector<int64_t>>(pads)),
            gert::InfershapeContextPara::OpAttr("ceil_mode", AnyValue::CreateFrom<bool>(false)),
            gert::InfershapeContextPara::OpAttr("data_format", AnyValue::CreateFrom<std::string>("NHWC")),
        });
}

TEST(Dilation2DBackpropInputAttrNarrowing, valid_control)
{
    auto context = MakeBaseCase({1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, "VALID", 3, 3);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 4, 4, 1}});
}

TEST(Dilation2DBackpropInputAttrNarrowing, large_positive_stride_must_not_wrap_to_one)
{
    // A positive stride of 2^32+1 is accepted by the int64 source check. With 64-bit arithmetic,
    // VALID output is 1x1, so out_backprop=3x3 must be rejected.
    auto context = MakeBaseCase({1, 4294967297LL, 4294967297LL, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, "VALID", 3, 3);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Dilation2DBackpropInputAttrNarrowing, large_positive_pad_must_not_wrap_to_one)
{
    // With CALCULATED padding, a 2^32+1 top/left pad should produce a much larger output than 3x3.
    // The full 64-bit value must be used for the output-shape check.
    auto context = MakeBaseCase({1, 1, 1, 1}, {1, 1, 1, 1}, {4294967297LL, 0, 4294967297LL, 0}, "CALCULATED", 4, 4);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

} // namespace
