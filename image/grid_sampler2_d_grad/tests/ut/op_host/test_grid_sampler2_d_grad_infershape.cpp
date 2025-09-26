/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grid_sampler2d_grad.cpp
 * \brief
 */
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "common/utils/ut_op_common.h"
#include "experiment_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"
#include "utils/op_desc_utils.h"
#include "utils/op_desc_utils_ex.h"

class grid_sampler2d_grad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grid_sampler2d_grad SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grid_sampler2d_grad TearDown" << std::endl;
    }
};

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fp16)
{
    ge::op::GridSampler2DGrad op;
    int n = 3;
    int c = 4;
    int h_in = 4;
    int w_in = 5;
    int h_out = 5;
    int w_out = 6;

    ge::DataType dtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in, w_in}, dtype, format, {n, c, h_in, w_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out, 2}, dtype, format, {n, h_out, w_out, 2}, format));
    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_SUCCESS);

    auto dx_desc = op.GetOutputDescByName("dx");
    // EXPECT_EQ(dx_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dx_shape = {n, c, h_in, w_in};
    EXPECT_EQ(dx_desc.GetShape().GetDims(), out_dx_shape);

    auto dgrid_desc = op.GetOutputDescByName("dgrid");
    // EXPECT_EQ(dgrid_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dgrid_shape = {n, h_out, w_out, 2};
    EXPECT_EQ(dgrid_desc.GetShape().GetDims(), out_dgrid_shape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fp32)
{
    ge::op::GridSampler2DGrad op;
    int n = 13;
    int c = 24;
    int h_in = 14;
    int w_in = 25;
    int h_out = 15;
    int w_out = 26;

    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in, w_in}, dtype, format, {n, c, h_in, w_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out, 2}, dtype, format, {n, h_out, w_out, 2}, format));

    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_SUCCESS);

    auto dx_desc = op.GetOutputDescByName("dx");
    EXPECT_EQ(dx_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dx_shape = {n, c, h_in, w_in};
    EXPECT_EQ(dx_desc.GetShape().GetDims(), out_dx_shape);

    auto dgrid_desc = op.GetOutputDescByName("dgrid");
    EXPECT_EQ(dgrid_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dgrid_shape = {n, h_out, w_out, 2};
    EXPECT_EQ(dgrid_desc.GetShape().GetDims(), out_dgrid_shape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_double)
{
    ge::op::GridSampler2DGrad op;
    int n = 13;
    int c = 24;
    int h_in = 14;
    int w_in = 25;
    int h_out = 15;
    int w_out = 26;

    ge::DataType dtype = ge::DT_DOUBLE;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in, w_in}, dtype, format, {n, c, h_in, w_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out, 2}, dtype, format, {n, h_out, w_out, 2}, format));

    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_SUCCESS);

    auto dx_desc = op.GetOutputDescByName("dx");
    // EXPECT_EQ(dx_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dx_shape = {n, c, h_in, w_in};
    EXPECT_EQ(dx_desc.GetShape().GetDims(), out_dx_shape);

    auto dgrid_desc = op.get_output_desc_dgrid();
    // EXPECT_EQ(dgrid_desc.GetDataType(), dtype);
    std::vector<int64_t> out_dgrid_shape = {n, h_out, w_out, 2};
    EXPECT_EQ(dgrid_desc.GetShape().GetDims(), out_dgrid_shape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_x)
{
    ge::op::GridSampler2DGrad op;
    int n = 1;
    int c = 2;
    int h_in = 4;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in}, dtype, format, {n, c, h_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out, 2}, dtype, format, {n, h_out, w_out, 2}, format));
    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_FAILED);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_grad)
{
    ge::op::GridSampler2DGrad op;
    int n = 13;
    int c = 24;
    int h_in = 54;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    ge::DataType dtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, h_out, w_out}, dtype, format, {n, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in, w_in}, dtype, format, {n, c, h_in, w_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out, 2}, dtype, format, {n, h_out, w_out, 2}, format));
    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_FAILED);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_grid)
{
    ge::op::GridSampler2DGrad op;
    int n = 13;
    int c = 24;
    int h_in = 54;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    ge::DataType dtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
    op.UpdateInputDesc("x", create_desc_with_ori({n, c, h_in, w_in}, dtype, format, {n, c, h_in, w_in}, format));
    op.UpdateInputDesc("grid", create_desc_with_ori({n, h_out, w_out}, dtype, format, {n, h_out, w_out}, format));
    Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners"}};
    EXPECT_EQ(InferShapeTest(op, param1), ge::GRAPH_FAILED);
}