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

#include "../../../src/framework/affine_grid_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxAffineGridPluginTest, ParseAlignCorners)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"align_corners","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("affine_grid");
    bool align_corners = false;

    EXPECT_EQ(domi::ParseParamsAffineGrid(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(align_corners);
}

TEST(OnnxAffineGridPluginTest, UseDefaultAlignCornersWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("affine_grid");
    bool align_corners = true;

    EXPECT_EQ(domi::ParseParamsAffineGrid(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(align_corners);
}

TEST(OnnxAffineGridPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("affine_grid");

    EXPECT_EQ(domi::ParseParamsAffineGrid(op_src, op_dest), domi::FAILED);
}
