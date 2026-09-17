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
#include <limits>

#include <gtest/gtest.h>

#include "../../../../op_host/op_api/aclnn_roi_align_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int64_t kWraparoundValue = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 2;

aclnnStatus RunRoiAlignV2(int64_t pooledHeight, int64_t pooledWidth, int64_t numBoxes)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({numBoxes, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({numBoxes, 1, pooledHeight, pooledWidth}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, pooledHeight, pooledWidth, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

} // namespace

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, accepts_valid_control) { EXPECT_EQ(RunRoiAlignV2(2, 2, 1), ACL_SUCCESS); }

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, accepts_int32_max_height_boundary)
{
    EXPECT_EQ(RunRoiAlignV2(kInt32Max, 2, 0), ACL_SUCCESS);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, accepts_int32_max_width_boundary)
{
    EXPECT_EQ(RunRoiAlignV2(2, kInt32Max, 0), ACL_SUCCESS);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, rejects_height_int32_max_plus_one)
{
    EXPECT_EQ(RunRoiAlignV2(kInt32Max + 1, 2, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, rejects_width_int32_max_plus_one)
{
    EXPECT_EQ(RunRoiAlignV2(2, kInt32Max + 1, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, rejects_height_wraparound)
{
    EXPECT_EQ(RunRoiAlignV2(kWraparoundValue, 2, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingApi, rejects_width_wraparound)
{
    EXPECT_EQ(RunRoiAlignV2(2, kWraparoundValue, 0), ACLNN_ERR_PARAM_INVALID);
}
