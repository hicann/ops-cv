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
#include "image/upsample_bilinear2d_aa_backward/op_host/op_api/aclnn_upsample_bilinear2d_aa_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsample_bilinear2d_aa_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_upsample_bilinear2d_aa_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_upsample_bilinear2d_aa_backward_test TearDown" << std::endl;
    }
};

// gradOutput nullptr
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_gradOutput_nullptr)
{
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_out_nullptr)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// inputsize nullptr
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_outputsize_nullptr)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, nullptr, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float16
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_dtype_float16)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// dtype float32
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_dtype_float32)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// gradOutput empty N
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_gradOutput_empty_n)
{
    auto gradOutput_desc = TensorDesc({0, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput empty C
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_gradOutput_empty_c)
{
    auto gradOutput_desc = TensorDesc({1, 0, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput empty H
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_gradOutput_empty_h)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{0, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputsize value 0
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_inputsize_value_0)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 0});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputsize size 5
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_inputsize_size_5)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput shape 3d
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_shape_3d)
{
    auto gradOutput_desc = TensorDesc({1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput and out dtype different
TEST_F(l2_upsample_bilinear2d_aa_backward_test, l2_upsample_bilinear2d_aa_backward_test_dtype_different)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dAABackward,
        INPUT(gradOutput_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
