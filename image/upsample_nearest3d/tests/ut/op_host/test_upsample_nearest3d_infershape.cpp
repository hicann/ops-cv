/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "image_ops.h"
#include "common/utils/ut_op_common.h"

class UpsampleNearest3dTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleNearest3dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleNearest3dTest TearDown" << std::endl;
    }
};

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test1_failed)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_INT8, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test2_failed)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test3_failed)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    std::vector<int64_t> output_size_vec = {10, 10};
    op.SetAttr("output_size", output_size_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test4_failed)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    std::vector<float> scales_vec = {2.0, 2.0};
    op.SetAttr("scales", scales_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test5_failed)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    std::vector<int64_t> output_size_vec = {10, 10, 10};
    op.SetAttr("output_size", output_size_vec);
    std::vector<float> scales_vec = {2.0, 2.0, 2.0};
    op.SetAttr("scales", scales_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test1_success)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    std::vector<int64_t> output_size_vec = {10, 10, 10};
    op.SetAttr("output_size", output_size_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_var_output_shape = {1, 1, 10, 10, 10};
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test2_success)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc(
        "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
    std::vector<float> scales_vec = {2.0, 2.0, 2.0};
    op.SetAttr("scales", scales_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_var_output_shape = {1, 1, 10, 10, 10};
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test3_success)
{
    ge::op::UpsampleNearest3d op;
    op.UpdateInputDesc("x",
        create_desc_with_ori({16, 3, 5, 512, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {16, 3, 5, 512, 512}, ge::FORMAT_ND));
    std::vector<float> scales_vec = {2.5, 2.0, 3.8};
    op.SetAttr("scales", scales_vec);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_var_output_shape = {16, 3, 12, 1024, 1945};
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

    Runtime2TestParam param;
    param.attrs = {"output_size", "scales"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    out_var_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
