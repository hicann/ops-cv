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
#include "../../../../op_host/op_api/aclnn_upsample_nearest_1d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsampleNearest1d_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_upsampleNearest1d_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_upsampleNearest1d_backward_test TearDown" << std::endl;
    }
};

// // 支持float32
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_001)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {4};
    vector<int64_t> input_size = {1, 1, 2};

    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 支持float16
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_002)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {4};
    vector<int64_t> input_size = {1, 1, 2};

    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持int64
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_007)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_INT64, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_INT64, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int16
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_008)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_INT16, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_INT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持bool
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_010)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_BOOL, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_BOOL, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex64
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_011)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_COMPLEX64, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_COMPLEX64, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex128
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_012)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_COMPLEX128, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_COMPLEX128, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持空tensor
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_013)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {0, 1, 3};

    auto self_desc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({0, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持shape为2的输入
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_014)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为5的输入
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_015)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持空指针
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_016)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(nullptr, output_size_desc, input_size_desc, scales), OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持输入和输出的dtype不一致
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_017)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的size大小为3
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_018)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2, 2, 2};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的元素值小于1
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_019)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {0};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的size大小为5
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_020)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的元素值小于1
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_021)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {0, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// align_corners为True
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_022)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {0, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和inputsize在nc维度上不一样
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_023)
{
    const double_t scales = 2.0;

    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {2, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和outputsize在hw维度上不一样
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_024)
{
    const double_t scales = 2.0;
    vector<int64_t> output_size = {4};
    vector<int64_t> input_size = {1, 1, 3};

    auto self_desc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// scales校验
TEST_F(l2_upsampleNearest1d_backward_test, l2_upsampleNearest1d_backward_test_025)
{
    const double_t scales = 1.0;
    vector<int64_t> output_size = {10};
    vector<int64_t> input_size = {1, 1, 5};

    auto self_desc = TensorDesc({1, 1, 10}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnUpsampleNearest1dBackward, INPUT(self_desc, output_size_desc, input_size_desc, scales),
        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}