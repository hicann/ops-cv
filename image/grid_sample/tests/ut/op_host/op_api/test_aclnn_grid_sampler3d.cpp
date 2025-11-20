/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grid_sampler3d.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_grid_sampler3d_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "grid_sampler3d_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "grid_sampler3d_test TearDown" << std::endl; }
};

// input nullptr
TEST_F(l2_grid_sampler3d_test, input_nullptr) {
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(nullptr, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grid nullptr
TEST_F(l2_grid_sampler3d_test, grid_nullptr) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D,INPUT(inputDesc, nullptr, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_grid_sampler3d_test, out_nullptr) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(nullptr));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// empty tensor
TEST_F(l2_grid_sampler3d_test, input_empty) {
  auto inputDesc = TensorDesc({2, 0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float16
TEST_F(l2_grid_sampler3d_test, dtype_float16) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float32
TEST_F(l2_grid_sampler3d_test, dtype_float32) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// invalid dtype int8
TEST_F(l2_grid_sampler3d_test, dtype_int8) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_INT8, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// valid dtype bf16
TEST_F(l2_grid_sampler3d_test, dtype_bf16) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// invalid shape n
TEST_F(l2_grid_sampler3d_test, shape_n_input1_grid2) {
  auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape c
TEST_F(l2_grid_sampler3d_test, shape_c_input1_out2) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 3, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape h
TEST_F(l2_grid_sampler3d_test, shape_h_grid2_out3) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape w
TEST_F(l2_grid_sampler3d_test, shape_w_grid2_out4) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape grid
TEST_F(l2_grid_sampler3d_test, shape_grid_not_3) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid mode
TEST_F(l2_grid_sampler3d_test, mode_3) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 3, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid padding mode
TEST_F(l2_grid_sampler3d_test, padding_mode_3) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 3, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input invalid format
TEST_F(l2_grid_sampler3d_test, input_format_invalid) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out invalid fotmat
TEST_F(l2_grid_sampler3d_test, out_format_invalid) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grid_sampler3d_test, ascend910B2_case_01) {
  auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grid_sampler3d_test, isNCDHW_special_case) {
  auto inputDesc = TensorDesc({22, 4, 16, 64, 64}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-100, 100);
  auto gridDesc = TensorDesc({22, 16, 64, 64, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({22, 4, 16, 64, 64}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-100, 100);
  auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}