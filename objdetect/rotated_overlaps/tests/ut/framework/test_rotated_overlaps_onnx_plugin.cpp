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

#include "../../../framework/npu_rotated_overlaps_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuRotatedOverlapsPluginTest, ParseTransAttribute)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"trans","type":2,"i":1}]})",
                                               "rotated_overlaps_src");
    ge::Operator op_dest = CreateOperator("rotated_overlaps");
    bool trans = false;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsNpuRotatedOverlaps(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(trans);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "rotated_overlaps_src");
}

TEST(OnnxNpuRotatedOverlapsPluginTest, ParseZeroTransWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"trans","type":2}]})");
    ge::Operator op_dest = CreateOperator("rotated_overlaps");
    bool trans = true;

    EXPECT_EQ(domi::ParseParamsNpuRotatedOverlaps(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(trans);
}

TEST(OnnxNpuRotatedOverlapsPluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("rotated_overlaps");
    bool trans = true;

    EXPECT_EQ(domi::ParseParamsNpuRotatedOverlaps(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(trans);
}

TEST(OnnxNpuRotatedOverlapsPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("rotated_overlaps");
    EXPECT_EQ(domi::ParseParamsNpuRotatedOverlaps(op_src, op_dest), domi::FAILED);
}
