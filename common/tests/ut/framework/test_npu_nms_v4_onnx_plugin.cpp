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

#include <cstdint>
#include <cstring>
#include <string>

#include "../../../src/framework/npu_nms_v4_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuNmsV4PluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pad_to_max_output_size","type":2,"i":1},{"name":"max_output_size","type":2,"i":128}]})",
        "nmsv4_src");
    ge::Operator op_dest = CreateOperator("nms_v4");
    bool pad_to_max_output_size = false;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsNmsV4(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pad_to_max_output_size", pad_to_max_output_size), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(pad_to_max_output_size);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "nmsv4_src");
}

TEST(OnnxNpuNmsV4PluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pad_to_max_output_size","type":2},{"name":"max_output_size","type":2}]})");
    ge::Operator op_dest = CreateOperator("nms_v4");
    bool pad_to_max_output_size = true;
    ge::Tensor max_output_size;

    EXPECT_EQ(domi::ParseParamsNmsV4(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pad_to_max_output_size", pad_to_max_output_size), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(pad_to_max_output_size);
    EXPECT_EQ(op_dest.GetAttr("max_output_size", max_output_size), ge::GRAPH_SUCCESS);
    ASSERT_EQ(max_output_size.GetSize(), sizeof(int32_t));
    int32_t value = 1;
    std::memcpy(&value, max_output_size.GetData(), sizeof(value));
    EXPECT_EQ(value, 0);
}

TEST(OnnxNpuNmsV4PluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("nms_v4");
    bool pad_to_max_output_size = true;

    EXPECT_EQ(domi::ParseParamsNmsV4(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pad_to_max_output_size", pad_to_max_output_size), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(pad_to_max_output_size);
}

TEST(OnnxNpuNmsV4PluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("nms_v4");
    EXPECT_EQ(domi::ParseParamsNmsV4(op_src, op_dest), domi::FAILED);
}
