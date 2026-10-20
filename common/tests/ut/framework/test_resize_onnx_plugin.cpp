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

#include <string>
#include <vector>

#include "../../../src/framework/resize_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, int input_num = 3,
                                  const std::vector<ge::AscendString>& onnx_inputs = {})
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    if (input_num > 0) {
        op_src.DynamicInputRegister("x", input_num);
    }
    if (!onnx_inputs.empty()) {
        op_src.SetAttr("input", onnx_inputs);
    }
    return op_src;
}
} // namespace

TEST(OnnxResizePluginTest, ParseParamsResizeAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"coordinate_transformation_mode","type":3,"s":"align_corners"},)"
        R"({"name":"mode","type":3,"s":"linear"}]})",
        3);
    ge::Operator op_dest = CreateOperator("resize");
    ge::AscendString mode;
    ge::AscendString coord_mode;
    int64_t input_size = 0;

    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "linear");
    EXPECT_EQ(op_dest.GetAttr("coordinate_transformation_mode", coord_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(coord_mode.GetString(), "align_corners");
    EXPECT_EQ(op_dest.GetAttr("input_size", input_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(input_size, 3);
}

TEST(OnnxResizePluginTest, ParseAntialiasWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"antialias","type":2}]})", 3);
    ge::Operator op_dest = CreateOperator("resize");
    ge::AscendString mode;
    ge::AscendString coord_mode;

    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "nearest");
    EXPECT_EQ(op_dest.GetAttr("coordinate_transformation_mode", coord_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(coord_mode.GetString(), "half_pixel");
}

TEST(OnnxResizePluginTest, ParseParamsResizeDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})", 3);
    ge::Operator op_dest = CreateOperator("resize");
    ge::AscendString mode;
    ge::AscendString coord_mode;

    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "nearest");
    EXPECT_EQ(op_dest.GetAttr("coordinate_transformation_mode", coord_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(coord_mode.GetString(), "half_pixel");
}

TEST(OnnxResizePluginTest, ParseParamsResizeEmptyFourthInputBecomesThree)
{
    std::vector<ge::AscendString> onnx_inputs = {ge::AscendString("X"), ge::AscendString("roi"),
                                                 ge::AscendString("scales"), ge::AscendString("")};
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})", 4, onnx_inputs);
    ge::Operator op_dest = CreateOperator("resize");
    int64_t input_size = 0;
    ge::AscendString input_roi;
    ge::AscendString input_scales;

    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("input_size", input_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(input_size, 3);
    EXPECT_EQ(op_dest.GetAttr("input_roi", input_roi), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(input_roi.GetString(), "roi");
    EXPECT_EQ(op_dest.GetAttr("input_scales", input_scales), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(input_scales.GetString(), "scales");
}

TEST(OnnxResizePluginTest, ParseParamsResizeFailWhenInputsInsufficient)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})", 2);
    ge::Operator op_dest = CreateOperator("resize");
    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::FAILED);
}

TEST(OnnxResizePluginTest, ParseParamsResizeReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{", 3);
    ge::Operator op_dest = CreateOperator("resize");
    EXPECT_EQ(domi::ParseParamsResize(op_src, op_dest), domi::FAILED);
}

TEST(OnnxResizePluginTest, ParseParamsResizeV10Mode)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"mode","type":3,"s":"linear"}]})", 0);
    ge::Operator op_dest = CreateOperator("resize_v10");
    ge::AscendString mode;

    EXPECT_EQ(domi::ParseParamsResizeV10(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "linear");
}

TEST(OnnxResizePluginTest, ParseParamsResizeV10DefaultMode)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})", 0);
    ge::Operator op_dest = CreateOperator("resize_v10");
    ge::AscendString mode;

    EXPECT_EQ(domi::ParseParamsResizeV10(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "nearest");
}

TEST(OnnxResizePluginTest, ParseParamsResizeV10ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{", 0);
    ge::Operator op_dest = CreateOperator("resize_v10");
    EXPECT_EQ(domi::ParseParamsResizeV10(op_src, op_dest), domi::FAILED);
}
