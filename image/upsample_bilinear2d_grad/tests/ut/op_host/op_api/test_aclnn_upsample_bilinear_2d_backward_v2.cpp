/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_upsample_bilinear_2d_backward_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsample_bilinear2d_backward_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_upsample_bilinear2d_backward_v2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_upsample_bilinear2d_backward_v2_test TearDown" << std::endl;
    }
};

// 支持float32
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_001)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 支持float16
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_002)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持double
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_003)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int8
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_004)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_INT8, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持uint8
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_005)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_UINT8, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int32
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_006)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_INT32, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int64
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_007)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_INT64, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int16
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_008)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_INT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持bool
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_010)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_BOOL, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex64
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_011)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_COMPLEX64, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex128
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_012)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_COMPLEX128, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持空tensor
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_013)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({0, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持shape为2的输入
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_014)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为5的输入
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_015)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持空指针
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_016)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持输入和输出的dtype不一致
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_017)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的size大小为3
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_018)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的元素值小于1
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_019)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {0, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的size大小为5
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_020)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的元素值小于1
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_021)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// align_corners为True
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_022)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和inputsize在nc维度上不一样
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_023)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {2, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和outputsize在hw维度上不一样
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_024)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2, 4};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_025)
{
    const double_t scales_h = 120.0;
    const double_t scales_w = 120.0;
    bool align_corners = false;
    vector<int64_t> output_size = {360, 360};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 360, 360}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_026)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持空指针
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_028)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持空指针
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_029)
{
    const double_t scales_h = 120.0;
    const double_t scales_w = 120.0;
    bool align_corners = false;
    vector<int64_t> output_size = {360, 360};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 360, 360}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_030)
{
    const double_t scales_h = 120.0;
    const double_t scales_w = 120.0;
    bool align_corners = false;
    vector<int64_t> output_size = {360, 360};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 360, 360}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_bilinear2d_backward_v2_test, Ascend910B2_upsample_bilinear2d_backward_v2_test_031)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 输出的格式不在支持范围内
TEST_F(l2_upsample_bilinear2d_backward_v2_test, l2_upsample_bilinear2d_backward_v2_test_032)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// NCHW格式下，scales_h不满足约束条件
TEST_F(l2_upsample_bilinear2d_backward_v2_test, Ascend910B2_l2_upsample_bilinear2d_backward_v2_test_033)
{
    const double_t scales_h = 4.0;
    const double_t scales_w = 4.0;
    bool align_corners = false;
    vector<int64_t> output_size = {5, 5};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// NCHW格式下，scales_w不满足约束条件
TEST_F(l2_upsample_bilinear2d_backward_v2_test, Ascend910B2_l2_upsample_bilinear2d_backward_v2_test_034)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// NHWC格式下，scales_w不满足约束条件
TEST_F(l2_upsample_bilinear2d_backward_v2_test, Ascend910B2_l2_upsample_bilinear2d_backward_v2_test_035)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 5.0;
    bool align_corners = false;
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> input_size = {1, 3, 3, 1};

    auto self_desc = TensorDesc({1, 6, 6, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 3, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// NHWC格式下，scales_h不满足约束条件
TEST_F(l2_upsample_bilinear2d_backward_v2_test, Ascend910B2_l2_upsample_bilinear2d_backward_v2_test_036)
{
    const double_t scales_h = 4.0;
    const double_t scales_w = 4.0;
    bool align_corners = false;
    vector<int64_t> output_size = {8, 8};
    vector<int64_t> input_size = {1, 3, 3, 1};

    auto self_desc = TensorDesc({1, 8, 8, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 3, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackwardV2,
        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
