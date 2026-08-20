/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_api/aclnn_gaussian_blur.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class AclnnGaussianBlurTest : public testing::Test {
protected:
    static void SetUpTestCase() { op::SetPlatformSocVersion(op::SocVersion::ASCEND950); }

    static void TearDownTestCase() { op::SetPlatformSocVersion(op::SocVersion::ASCEND910B); }
};

TEST_F(AclnnGaussianBlurTest, accepts_float_nd_hwc)
{
    auto src = TensorDesc({8, 8, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 8 * 3, 1.0f));
    auto dst = TensorDesc({8, 8, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(8 * 8 * 3, 0.0f));
    auto ksize = IntArrayDesc(vector<int64_t>{3, 3});
    auto test = OP_API_UT(aclnnGaussianBlur, INPUT(src, ksize, 1.0, 1.0, 1, dst), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(AclnnGaussianBlurTest, rejects_unsupported_dtype_and_kernel)
{
    auto src = TensorDesc({8, 8, 3}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>(8 * 8 * 3, 1));
    auto dst = TensorDesc({8, 8, 3}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>(8 * 8 * 3, 0));
    auto ksize = IntArrayDesc(vector<int64_t>{4, 3});
    auto test = OP_API_UT(aclnnGaussianBlur, INPUT(src, ksize, 1.0, 1.0, 1, dst), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
