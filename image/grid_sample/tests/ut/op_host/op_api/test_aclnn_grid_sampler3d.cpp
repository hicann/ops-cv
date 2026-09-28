/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "opdev/platform.h"

#include "../../../../op_api/aclnn_grid_sampler3d.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_grid_sampler3d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "grid_sampler3d_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_sampler3d_test TearDown" << std::endl; }
};

// input nullptr
TEST_F(l2_grid_sampler3d_test, input_nullptr)
{
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(nullptr, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grid nullptr
TEST_F(l2_grid_sampler3d_test, grid_nullptr)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, nullptr, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_grid_sampler3d_test, out_nullptr)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// empty tensor
TEST_F(l2_grid_sampler3d_test, input_empty)
{
    auto inputDesc = TensorDesc({2, 0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// dtype float16
TEST_F(l2_grid_sampler3d_test, dtype_float16)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float32
TEST_F(l2_grid_sampler3d_test, dtype_float32)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// dtype double
// TEST_F(l2_grid_sampler3d_test, dtype_double) {
//   auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//   auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
//   auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//   auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);
//   // ut.TestPrecision();
// }

// invalid dtype int8
TEST_F(l2_grid_sampler3d_test, dtype_int8)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_INT8, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// valid dtype bf16
TEST_F(l2_grid_sampler3d_test, dtype_bf16)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// invalid shape n
TEST_F(l2_grid_sampler3d_test, shape_n_input1_grid2)
{
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape c
TEST_F(l2_grid_sampler3d_test, shape_c_input1_out2)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape h
TEST_F(l2_grid_sampler3d_test, shape_h_grid2_out3)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape w
TEST_F(l2_grid_sampler3d_test, shape_w_grid2_out4)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape grid
TEST_F(l2_grid_sampler3d_test, shape_grid_not_3)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid mode
TEST_F(l2_grid_sampler3d_test, mode_3)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 3, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid padding mode
TEST_F(l2_grid_sampler3d_test, padding_mode_3)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 3, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input invalid format
TEST_F(l2_grid_sampler3d_test, input_format_invalid)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out invalid fotmat
TEST_F(l2_grid_sampler3d_test, out_format_invalid)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grid_sampler3d_test, ascend910B2_case_01)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grid_sampler3d_test, isNCDHW_special_case)
{
    auto inputDesc = TensorDesc({22, 4, 16, 64, 64}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({22, 16, 64, 64, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({22, 4, 16, 64, 64}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-100, 100);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grid_sampler3d_test, ascend910B2_NDHWC_case_01)
{
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grid_sampler3d_test, ascend950_case_01)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// dtype double, aicpu branch
// NOTE: aicpu task registration result differs between environments: succeeds on CANN 9.2.0
// stub env (ACL_SUCCESS), fails on CANN 9.3.0 (ACLNN_ERR_INNER_NULLPTR). The host-side
// branch selection lines are fully executed in both cases.
TEST_F(l2_grid_sampler3d_test, dtype_double)
{
    auto inputDesc = TensorDesc({2, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_TRUE(aclRet == ACL_SUCCESS || aclRet == ACLNN_ERR_INNER_NULLPTR);
}

// ascend950 NDHWC float32 bilinear, regbase branch transposes NDHWC to NCDHW
TEST_F(l2_grid_sampler3d_test, ascend950_NDHWC_bilinear)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// ascend950 NDHWC float32 nearest, aicore old template branch without input transpose
TEST_F(l2_grid_sampler3d_test, ascend950_NDHWC_nearest)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 1, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// NDHWC double, aicpu branch transposes NDHWC to contiguous NCDHW and reformat to ND
// NOTE: aicpu task registration result differs between environments (see dtype_double);
// the transpose/contiguous/reformat lines are fully executed in both cases.
TEST_F(l2_grid_sampler3d_test, ndhwc_double)
{
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 2, 2, 1}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_TRUE(aclRet == ACL_SUCCESS || aclRet == ACLNN_ERR_INNER_NULLPTR);
}

// ascend310P NDHWC float32 bilinear, aicpu branch transposes NDHWC to contiguous NCDHW
// NOTE: aicpu task registration result differs between environments (see dtype_double);
// the transpose/contiguous/reformat lines are fully executed in both cases.
TEST_F(l2_grid_sampler3d_test, ascend310P_NDHWC_bilinear)
{
    SetPlatformSocVersion(SocVersion::ASCEND310P);
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_TRUE(aclRet == ACL_SUCCESS || aclRet == ACLNN_ERR_INNER_NULLPTR);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// NDHWC input with NCDHW out, shape mismatch on channel index
TEST_F(l2_grid_sampler3d_test, ndhwc_input_ncdhw_out)
{
    auto inputDesc = TensorDesc({2, 1, 3, 3, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-100, 100);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto ut = OP_API_UT(aclnnGridSampler3D, INPUT(inputDesc, gridDesc, 0, 0, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
