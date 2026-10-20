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

#include "../../../framework/roi_align_rotated_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}

ge::Operator CreateSourceOperatorWithIo(const std::string& attrs, uint32_t input_num, uint32_t output_num,
                                        const std::string& name = "src")
{
    ge::Operator op_src = CreateSourceOperator(attrs, name);
    op_src.DynamicInputRegister("x", input_num);
    op_src.DynamicOutputRegister("y", output_num);
    return op_src;
}
} // namespace

TEST(OnnxRoiAlignRotatedPluginTest, ParseRequiredAttributes)
{
    ge::Operator op_src = CreateSourceOperatorWithIo(
        R"({"attribute":[{"name":"output_height","type":2,"i":7},{"name":"output_width","type":2,"i":7},)"
        R"({"name":"spatial_scale","type":1,"f":"0.25"},{"name":"aligned","type":2,"i":0},)"
        R"({"name":"clockwise","type":2,"i":1},{"name":"sampling_ratio","type":2,"i":2}]})",
        2, 1, "roi_rot_src");
    ge::Operator op_dest = CreateOperator("roi_align_rotated");
    int64_t pooled_h = 0;
    int64_t pooled_w = 0;
    float spatial_scale = 0.0f;
    int64_t sampling_ratio = 0;
    bool aligned = true;
    bool clockwise = false;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsRoiAlignRotated(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_h", pooled_h), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_h, 7);
    EXPECT_EQ(op_dest.GetAttr("pooled_w", pooled_w), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_w, 7);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.25f);
    EXPECT_EQ(op_dest.GetAttr("sampling_ratio", sampling_ratio), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sampling_ratio, 2);
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(aligned);
    EXPECT_EQ(op_dest.GetAttr("clockwise", clockwise), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(clockwise);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "roi_rot_src");
}

TEST(OnnxRoiAlignRotatedPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperatorWithIo(
        R"({"attribute":[{"name":"output_height","type":2},{"name":"output_width","type":2},)"
        R"({"name":"spatial_scale","type":1},{"name":"aligned","type":2},)"
        R"({"name":"clockwise","type":2},{"name":"sampling_ratio","type":2}]})",
        2, 1);
    ge::Operator op_dest = CreateOperator("roi_align_rotated");
    int64_t pooled_h = 9;
    int64_t pooled_w = 9;
    float spatial_scale = 1.0f;
    int64_t sampling_ratio = 9;
    bool aligned = true;
    bool clockwise = true;

    EXPECT_EQ(domi::ParseParamsRoiAlignRotated(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_h", pooled_h), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_h, 0);
    EXPECT_EQ(op_dest.GetAttr("pooled_w", pooled_w), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_w, 0);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("sampling_ratio", sampling_ratio), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sampling_ratio, 0);
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(aligned);
    EXPECT_EQ(op_dest.GetAttr("clockwise", clockwise), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(clockwise);
}

TEST(OnnxRoiAlignRotatedPluginTest, KeepOptionalDefaultsWhenAttrsAbsent)
{
    ge::Operator op_src = CreateSourceOperatorWithIo(
        R"({"attribute":[{"name":"output_height","type":2,"i":3},{"name":"output_width","type":2,"i":4},)"
        R"({"name":"spatial_scale","type":1,"f":"0.5"}]})",
        2, 1);
    ge::Operator op_dest = CreateOperator("roi_align_rotated");
    int64_t sampling_ratio = 9;
    bool aligned = false;
    bool clockwise = true;

    EXPECT_EQ(domi::ParseParamsRoiAlignRotated(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("sampling_ratio", sampling_ratio), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sampling_ratio, 0);
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(aligned);
    EXPECT_EQ(op_dest.GetAttr("clockwise", clockwise), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(clockwise);
}

TEST(OnnxRoiAlignRotatedPluginTest, ReturnFailedWhenRequiredAttrsMissing)
{
    ge::Operator op_src = CreateSourceOperatorWithIo(R"({"attribute":[]})", 2, 1);
    ge::Operator op_dest = CreateOperator("roi_align_rotated");
    EXPECT_EQ(domi::ParseParamsRoiAlignRotated(op_src, op_dest), domi::FAILED);
}

TEST(OnnxRoiAlignRotatedPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("roi_align_rotated");
    EXPECT_EQ(domi::ParseParamsRoiAlignRotated(op_src, op_dest), domi::FAILED);
}
