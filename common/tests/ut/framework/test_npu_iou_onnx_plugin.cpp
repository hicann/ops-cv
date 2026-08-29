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

#include "../../../src/framework/npu_iou_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxIouPluginTest, ParseModeIof)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"mode","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("iou");
    ge::AscendString mode;
    float eps = 0.0f;
    bool aligned = true;

    EXPECT_EQ(domi::ParseParamsNpuIou(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "iof");
    EXPECT_EQ(op_dest.GetAttr("eps", eps), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(eps, 0.01f);
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(aligned);
}

TEST(OnnxIouPluginTest, UseDefaultModeIouWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("iou");
    ge::AscendString mode;

    EXPECT_EQ(domi::ParseParamsNpuIou(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "iou");
}

TEST(OnnxIouPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("iou");

    EXPECT_EQ(domi::ParseParamsNpuIou(op_src, op_dest), domi::FAILED);
}
