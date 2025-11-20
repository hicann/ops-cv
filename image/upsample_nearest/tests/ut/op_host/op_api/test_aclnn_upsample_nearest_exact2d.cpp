/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_upsample_nearest_exact2d.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_nearest_exact2d_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_nearest_exact2d_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "upsample_nearest_exact2d_test TearDown" << endl;
    }
};

TEST_F(l2_upsample_nearest_exact2d_test, case_float_normal)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float_out_shape_success)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 0};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 0, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_bfloat16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 6}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_shape_6d_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_self_nullptr_abnormal)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnUpsampleNearestExact2d, INPUT((aclTensor*)nullptr, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float_nhwc)
{
    auto self_desc = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut =
        OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
