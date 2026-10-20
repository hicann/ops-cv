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

#include "../../../src/framework/nms_rotated_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNMSRotatedPluginTest, ParseIouThreshold)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"iou_threshold","type":1,"f":"0.7"}]})",
                                               "nms_rot_src");
    ge::Operator op_dest = CreateOperator("nms_rotated");
    float iou_threshold = 0.0f;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsNMSRotatedCall(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.7f);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "nms_rot_src");
}

TEST(OnnxNMSRotatedPluginTest, ParseZeroIouThresholdWithoutFField)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"iou_threshold","type":1}]})");
    ge::Operator op_dest = CreateOperator("nms_rotated");
    float iou_threshold = 1.0f;

    EXPECT_EQ(domi::ParseParamsNMSRotatedCall(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.0f);
}

TEST(OnnxNMSRotatedPluginTest, ReturnFailedWhenIouThresholdMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("nms_rotated");
    EXPECT_EQ(domi::ParseParamsNMSRotatedCall(op_src, op_dest), domi::FAILED);
}

TEST(OnnxNMSRotatedPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("nms_rotated");
    EXPECT_EQ(domi::ParseParamsNMSRotatedCall(op_src, op_dest), domi::FAILED);
}
