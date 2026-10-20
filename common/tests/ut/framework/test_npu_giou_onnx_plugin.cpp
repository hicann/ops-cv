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

#include "../../../src/framework/npu_giou_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuGiouPluginTest, ParseAllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"trans","type":2,"i":1},{"name":"is_cross","type":2,"i":1},)"
        R"({"name":"mode","type":2,"i":1}]})",
        "giou_src");
    ge::Operator op_dest = CreateOperator("giou");
    bool trans = false;
    bool is_cross = false;
    ge::AscendString mode;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsNpuGiou(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(trans);
    EXPECT_EQ(op_dest.GetAttr("is_cross", is_cross), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(is_cross);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "iof");
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "giou_src");
}

TEST(OnnxNpuGiouPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"trans","type":2},{"name":"is_cross","type":2},{"name":"mode","type":2}]})");
    ge::Operator op_dest = CreateOperator("giou");
    bool trans = true;
    bool is_cross = true;
    ge::AscendString mode;

    EXPECT_EQ(domi::ParseParamsNpuGiou(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(trans);
    EXPECT_EQ(op_dest.GetAttr("is_cross", is_cross), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(is_cross);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "iou");
}

TEST(OnnxNpuGiouPluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("giou");
    bool trans = true;
    ge::AscendString mode;

    EXPECT_EQ(domi::ParseParamsNpuGiou(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("trans", trans), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(trans);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "iou");
}

TEST(OnnxNpuGiouPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("giou");
    EXPECT_EQ(domi::ParseParamsNpuGiou(op_src, op_dest), domi::FAILED);
}
