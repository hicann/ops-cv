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

#include "../../../src/framework/psroi_poolingV2_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxPSROIPoolingV2PluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1,"f":"0.25"},{"name":"output_dim","type":2,"i":8},{"name":"group_size","type":2,"i":4}]})");
    ge::Operator op_dest = CreateOperator("psroi_pooling_v2");
    float spatial_scale = 0.0f;
    int output_dim = 0;
    int group_size = 0;

    EXPECT_EQ(domi::ParsePSROIPoolingV2Attributes(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.25f);
    EXPECT_EQ(op_dest.GetAttr("output_dim", output_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_dim, 8);
    EXPECT_EQ(op_dest.GetAttr("group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 4);
}

TEST(OnnxPSROIPoolingV2PluginTest, ParseZeroValuesWithoutScalarValueFields)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1},{"name":"output_dim","type":2},{"name":"group_size","type":2}]})");
    ge::Operator op_dest = CreateOperator("psroi_pooling_v2");
    float spatial_scale = -1.0f;
    int output_dim = -1;
    int group_size = -1;

    EXPECT_EQ(domi::ParsePSROIPoolingV2Attributes(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("output_dim", output_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_dim, 0);
    EXPECT_EQ(op_dest.GetAttr("group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 0);
}

TEST(OnnxPSROIPoolingV2PluginTest, ReturnFailedWhenRequiredAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1,"f":"0.25"},{"name":"output_dim","type":2,"i":8}]})");
    ge::Operator op_dest = CreateOperator("psroi_pooling_v2");

    EXPECT_EQ(domi::ParsePSROIPoolingV2Attributes(op_src, op_dest), domi::FAILED);
}

TEST(OnnxPSROIPoolingV2PluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("psroi_pooling_v2");

    EXPECT_EQ(domi::ParsePSROIPoolingV2Attributes(op_src, op_dest), domi::FAILED);
}
