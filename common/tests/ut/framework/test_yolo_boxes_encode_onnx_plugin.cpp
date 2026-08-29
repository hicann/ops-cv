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

#include "../../../src/framework/yolo_boxes_encode_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxYoloBoxesEncodePluginTest, ParsePerformanceMode)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"performance_mode","type":3,"s":"high_performance"}]})");
    ge::Operator op_dest = CreateOperator("yolo_boxes_encode");
    ge::AscendString performance_mode;

    EXPECT_EQ(domi::ParseParamsYoloBoxesEncode(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("performance_mode", performance_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(performance_mode.GetString(), "high_performance");
}

TEST(OnnxYoloBoxesEncodePluginTest, UseDefaultPerformanceModeWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("yolo_boxes_encode");
    ge::AscendString performance_mode;

    EXPECT_EQ(domi::ParseParamsYoloBoxesEncode(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("performance_mode", performance_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(performance_mode.GetString(), "high_precision");
}

TEST(OnnxYoloBoxesEncodePluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("yolo_boxes_encode");

    EXPECT_EQ(domi::ParseParamsYoloBoxesEncode(op_src, op_dest), domi::FAILED);
}
