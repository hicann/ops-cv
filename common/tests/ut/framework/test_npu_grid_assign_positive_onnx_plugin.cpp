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

#include "../../../src/framework/npu_grid_assign_positive_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuGridAssignPositivePluginTest, ParseAllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pos_iou_thr","type":1,"f":"0.7"},{"name":"min_pos_iou","type":1,"f":"0.2"},)"
        R"({"name":"gt_max_assign_all","type":2,"i":0},{"name":"num_gts","type":2,"i":64}]})",
        "gap_src");
    ge::Operator op_dest = CreateOperator("grid_assign");
    float pos_iou_thr = 0.0f;
    float min_pos_iou = 1.0f;
    bool gt_max_assign_all = true;
    int num_gts = 0;

    EXPECT_EQ(domi::ParseParamsNpuGridAssignPositive(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pos_iou_thr", pos_iou_thr), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(pos_iou_thr, 0.7f);
    EXPECT_EQ(op_dest.GetAttr("min_pos_iou", min_pos_iou), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(min_pos_iou, 0.2f);
    EXPECT_EQ(op_dest.GetAttr("gt_max_assign_all", gt_max_assign_all), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(gt_max_assign_all);
    EXPECT_EQ(op_dest.GetAttr("num_gts", num_gts), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_gts, 64);
}

TEST(OnnxNpuGridAssignPositivePluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pos_iou_thr","type":1},{"name":"gt_max_assign_all","type":2},)"
        R"({"name":"num_gts","type":2}]})");
    ge::Operator op_dest = CreateOperator("grid_assign");
    float pos_iou_thr = 1.0f;
    float min_pos_iou = 1.0f;
    bool gt_max_assign_all = true;
    int num_gts = 1;

    EXPECT_EQ(domi::ParseParamsNpuGridAssignPositive(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pos_iou_thr", pos_iou_thr), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(pos_iou_thr, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("min_pos_iou", min_pos_iou), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(min_pos_iou, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("gt_max_assign_all", gt_max_assign_all), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(gt_max_assign_all);
    EXPECT_EQ(op_dest.GetAttr("num_gts", num_gts), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_gts, 0);
}

TEST(OnnxNpuGridAssignPositivePluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("grid_assign");
    float pos_iou_thr = 0.0f;
    float min_pos_iou = 1.0f;
    bool gt_max_assign_all = false;
    int num_gts = 0;

    EXPECT_EQ(domi::ParseParamsNpuGridAssignPositive(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pos_iou_thr", pos_iou_thr), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(pos_iou_thr, 0.5f);
    EXPECT_EQ(op_dest.GetAttr("min_pos_iou", min_pos_iou), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(min_pos_iou, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("gt_max_assign_all", gt_max_assign_all), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(gt_max_assign_all);
    EXPECT_EQ(op_dest.GetAttr("num_gts", num_gts), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_gts, 128);
}

TEST(OnnxNpuGridAssignPositivePluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("grid_assign");
    EXPECT_EQ(domi::ParseParamsNpuGridAssignPositive(op_src, op_dest), domi::FAILED);
}
