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
#include <iostream>

#include "any_value.h"
#include "image/dilation2_d_backprop_input/op_kernel/arch35/dilation2_d_backprop_input_tiling_data.h"
#include "tiling_case_executor.h"

namespace {

using Ops::Cv::AnyValue;
constexpr int64_t kTooLarge = 4294967297LL;

gert::TilingContextPara MakeBaseCase(const std::vector<int64_t>& strides, const std::vector<int64_t>& rates,
                                     const std::vector<int64_t>& pads, const std::string& paddingMode, int64_t outH,
                                     int64_t outW)
{
    static int compileInfo = 0;
    return gert::TilingContextPara(
        "Dilation2DBackpropInput",
        {
            {{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, outH, outW, 1}, {1, outH, outW, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("strides", AnyValue::CreateFrom<std::vector<int64_t>>(strides)),
            gert::TilingContextPara::OpAttr("rates", AnyValue::CreateFrom<std::vector<int64_t>>(rates)),
            gert::TilingContextPara::OpAttr("padding_mode", AnyValue::CreateFrom<std::string>(paddingMode)),
            gert::TilingContextPara::OpAttr("pads", AnyValue::CreateFrom<std::vector<int64_t>>(pads)),
            gert::TilingContextPara::OpAttr("ceil_mode", AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("data_format", AnyValue::CreateFrom<std::string>("NHWC")),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
}

bool RunTiling(const gert::TilingContextPara& contextPara, Dilation2DBackpropInputTilingData& data)
{
    TilingInfo info;
    const bool accepted = ExecuteTiling(contextPara, info);
    std::cout << "[ATTR_NARROWING_TILING] accepted=" << accepted;
    if (accepted) {
        if (info.tilingDataSize < sizeof(Dilation2DBackpropInputTilingData)) {
            ADD_FAILURE() << "tiling data is shorter than expected: " << info.tilingDataSize;
            return false;
        }
        data = *reinterpret_cast<const Dilation2DBackpropInputTilingData*>(info.tilingData.get());
        std::cout << " strideH=" << data.strideH << " strideW=" << data.strideW << " rateH=" << data.rateH
                  << " rateW=" << data.rateW << " padTop=" << data.padTop << " padLeft=" << data.padLeft
                  << " outputH=" << data.outputH << " outputW=" << data.outputW;
    }
    std::cout << std::endl;
    return accepted;
}

TEST(Dilation2DBackpropInputAttrNarrowingTiling, valid_control)
{
    TilingInfo info;
    auto context = MakeBaseCase({1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, "VALID", 3, 3);
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(Dilation2DBackpropInputAttrNarrowingTiling, large_positive_stride_must_be_rejected_or_preserved)
{
    Dilation2DBackpropInputTilingData data;
    auto context = MakeBaseCase({1, kTooLarge, kTooLarge, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, "VALID", 3, 3);
    // The full 64-bit stride produces a 1x1 output, so the 3x3 out_backprop shape must be rejected.
    const bool accepted = RunTiling(context, data);
    EXPECT_FALSE(accepted);
}

TEST(Dilation2DBackpropInputAttrNarrowingTiling, large_positive_rate_must_be_rejected_or_preserved)
{
    Dilation2DBackpropInputTilingData data;
    auto context = MakeBaseCase({1, 1, 1, 1}, {1, kTooLarge, kTooLarge, 1}, {0, 0, 0, 0}, "SAME", 4, 4);
    const bool accepted = RunTiling(context, data);
    ASSERT_TRUE(accepted);
    EXPECT_EQ(data.rateH, kTooLarge);
    EXPECT_EQ(data.rateW, kTooLarge);
}

TEST(Dilation2DBackpropInputAttrNarrowingTiling, large_positive_pad_must_be_rejected_or_preserved)
{
    Dilation2DBackpropInputTilingData data;
    auto context = MakeBaseCase({1, 1, 1, 1}, {1, 1, 1, 1}, {kTooLarge, 0, 0, 0}, "CALCULATED", 4, 3);
    // The full pad makes outputH much larger than 4 and must fail the out_backprop shape check.
    const bool accepted = RunTiling(context, data);
    EXPECT_FALSE(accepted);
}

} // namespace
