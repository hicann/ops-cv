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
#include "../../../../op_host/op_api/aclnn_upsample_nearest_2d_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsamplenearest2dv2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_upsamplenearest2dv2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_upsamplenearest2dv2_test TearDown" << std::endl;
    }
};

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_invalid_dtype_int32)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_INT32, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_invalid_dtype_int64)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_INT64, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_INT64, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_invalid_dtype_not_equal)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_input_nullptr)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(nullptr, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_output_nullptr)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_invalid_shape)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3, 6};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6, 12};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_invalid_format)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_empty_tensor)
{
    vector<int64_t> output_size = {2, 2};

    auto self_desc = TensorDesc({0, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_float16_01)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_float16_02)
{
    vector<int64_t> inp_dims = {3, 3, 64, 128};
    vector<int64_t> output_size = {128, 256};
    vector<int64_t> out = {3, 3, 128, 256};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, ascend910B2_l2_upsamplenearest2dv2_test_bfloat16)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_BF16, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_float_01)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_float_02)
{
    vector<int64_t> inp_dims = {3, 3, 64, 128};
    vector<int64_t> output_size = {128, 256};
    vector<int64_t> out = {3, 3, 128, 256};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_double_01)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_DOUBLE, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_double_02)
{
    vector<int64_t> inp_dims = {3, 3, 64, 128};
    vector<int64_t> output_size = {128, 256};
    vector<int64_t> out = {3, 3, 128, 256};

    auto self_desc = TensorDesc(inp_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_DOUBLE, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_uint8)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_UINT8, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_UINT8, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_float_transpose)
{
    vector<int64_t> inp_dims = {1, 1, 3, 4};
    vector<int64_t> output_size = {6, 8};
    vector<int64_t> out = {1, 1, 6, 8};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW, {12, 12, 3, 1}, 0, {1, 1, 3, 4});
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_output_size_is_3)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_output_size_has_zero)
{
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 0};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest2dv2_test, l2_upsamplenearest2dv2_test_input_dim_one_is_zero)
{
    vector<int64_t> inp_dims = {1, 0, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = -99.0;
    float scalesW = -99.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, ascend910B4_case_float_scale_01)
{
    vector<int64_t> inp_dims = {1, 1, 3, 36};
    vector<int64_t> output_size = {6, 72};
    vector<int64_t> out = {1, 1, 6, 72};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = 2.0;
    float scalesW = 2.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, ascend910B4_case_float_scale_02)
{
    vector<int64_t> inp_dims = {1, 1, 6, 6};
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = 0.5;
    float scalesW = 0.5;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, Ascend910_93_case_float_NHWC)
{
    vector<int64_t> inp_dims = {1, 6, 6, 1};
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> out = {1, 3, 3, 1};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NHWC);
    float scalesH = 0.5;
    float scalesW = 0.5;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, Ascend910_9591_case_float_scale_01)
{
    vector<int64_t> inp_dims = {1, 1, 3, 36};
    vector<int64_t> output_size = {6, 72};
    vector<int64_t> out = {1, 1, 6, 72};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = 2.0;
    float scalesW = 2.0;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, Ascend910_9591_case_float_scale_02)
{
    vector<int64_t> inp_dims = {1, 1, 6, 6};
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);
    float scalesH = 0.5;
    float scalesW = 0.5;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest2dv2_test, Ascend910_9591_case_float_NHWC)
{
    vector<int64_t> inp_dims = {1, 6, 6, 1};
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> out = {1, 3, 3, 1};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NHWC);
    float scalesH = 0.5;
    float scalesW = 0.5;

    auto ut =
        OP_API_UT(aclnnUpsampleNearest2dV2, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}