/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_upsample_linear_1d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsamplelinear1d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_upsamplelinear1d_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_upsamplelinear1d_test TearDown" << std::endl;
    }
};

// input nullptr
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_input_nullptr)
{
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(nullptr, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// output nullptr
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_output_nullptr)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// output size nullptr
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_output_size_nullptr)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, nullptr, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// input empty
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_input_empty_0)
{
    auto selfDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = 3.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// input empty
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_input_empty_1)
{
    auto selfDesc = TensorDesc({1, 0, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 0, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid dtype int32
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_dtype_int32)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_INT32, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_INT32, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid dtype int64
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_dtype_int64)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_INT64, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_INT64, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape 2
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_shape_2)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// size 256
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_size_256)
{
    auto selfDesc = TensorDesc({1, 3, 64}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 3, 256}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({256});
    const double_t scales_h = 4.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// dtype float16, aicore
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_dtype_float16)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype float32, aicore
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_out_shape)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({8});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// scale conflict with outputSize
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_scale_output_size_conflict)
{
    auto selfDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = 2.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// scale using default value
TEST_F(l2_upsamplelinear1d_test, l2_upsamplelinear1d_test_scale_default)
{
    auto selfDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = -1.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// scale using default value
TEST_F(l2_upsamplelinear1d_test, Ascend910B2_l2_upsamplelinear1d_test_scale_npu)
{
    auto selfDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    const double_t scales_h = -1.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplelinear1d_test, Ascend910B2_l2_upsamplelinear1d_test_size_256_npu)
{
    auto selfDesc = TensorDesc({1, 3, 64}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 3, 256}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({256});
    const double_t scales_h = 4.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplelinear1d_test, Ascend910B2_l2_upsamplelinear1d_test_fp16)
{
    auto selfDesc = TensorDesc({1, 3, 64}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 3, 256}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({256});
    const double_t scales_h = 4.0;
    bool align_corners = false;
    auto ut = OP_API_UT(aclnnUpsampleLinear1d, INPUT(selfDesc, sizeDesc, align_corners, scales_h), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}