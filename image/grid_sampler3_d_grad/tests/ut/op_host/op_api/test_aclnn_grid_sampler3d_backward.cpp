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

#include "../../../../op_host/op_api/aclnn_grid_sampler3d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_grid_sampler3d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grid_sampler3d_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "grid_sampler3d_backward_test TearDown" << std::endl;
    }
};

// gradOutput nullptr
TEST_F(l2_grid_sampler3d_backward_test, gradOutput_nullptr)
{
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(nullptr, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// input nullptr
TEST_F(l2_grid_sampler3d_backward_test, input_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, nullptr, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grid nullptr
TEST_F(l2_grid_sampler3d_backward_test, grid_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, nullptr, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// inputGrad nullptr
TEST_F(l2_grid_sampler3d_backward_test, inputGrad_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(nullptr, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gridGrad nullptr
TEST_F(l2_grid_sampler3d_backward_test, gridGrad_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float16
TEST_F(l2_grid_sampler3d_backward_test, dtype_float16)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float32
TEST_F(l2_grid_sampler3d_backward_test, dtype_float32)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// // dtype double
// TEST_F(l2_grid_sampler3d_backward_test, dtype_double)
// {
//     auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
//     auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
//     auto ut = OP_API_UT(
//         aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
//         OUTPUT(inputGradDesc, gridGradDesc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// aicore dtype bf16
TEST_F(l2_grid_sampler3d_backward_test, dtype_bf16)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// formart NCDHW
TEST_F(l2_grid_sampler3d_backward_test, ascend910B2_case_transpose)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// output_mask false false
TEST_F(l2_grid_sampler3d_backward_test, output_mask_false_false)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{false, false});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// invalid shape n
TEST_F(l2_grid_sampler3d_backward_test, shape_n_input1_grid2)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape c
TEST_F(l2_grid_sampler3d_backward_test, shape_c_input1_output2)
{
    auto gradOutputDesc = TensorDesc({1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape h
TEST_F(l2_grid_sampler3d_backward_test, shape_h_grid2_output3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape w
TEST_F(l2_grid_sampler3d_backward_test, shape_w_grid2_output4)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid mode
TEST_F(l2_grid_sampler3d_backward_test, mode_3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 3, 0, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid padding mode
TEST_F(l2_grid_sampler3d_backward_test, padding_mode_3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(
        aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 3, false, outputMaskDesc),
        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
