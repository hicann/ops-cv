/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_align_rotated_tiling.cpp
 * \brief RoiAlignRotated（arch22）tiling 单元测试——按 soc 选编独立入口形态
 *        （stack_group_points 先例）：经 TILING_DIR=arch22 随非 ascend950 构建链接 arch22 通用
 *        NHWC tiling；ascend950（arch35）用例见
 *        arch35/test_roi_align_rotated_tiling.cpp，两侧互不混编。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/arch22/roi_align_rotated_tiling_arch22.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace ge;
using namespace std;

class TilingForRoiAlignRotated : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TilingForRoiAlignRotated SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TilingForRoiAlignRotated TearDown" << std::endl; }
};

TEST_F(TilingForRoiAlignRotated, roi_align_rotated_tiling_0)
{
    optiling::RoiAlignRotatedCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "RoiAlignRotated",
        {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 2, 2, 8}, {8, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
            gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
            gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
            gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("clockwise", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "4294967296 8 270582939649 34359738400 34359738376 34359738376 34359738376 "
                              "4539628424389459968 8589934593 2 262144 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

static gert::TilingContextPara MakeRoiAlignRotatedTilingPara(uint32_t rois_num, int64_t pooled_h, int64_t pooled_w)
{
    static optiling::RoiAlignRotatedCompileInfo compileInfo = {48, 196608};
    int64_t r = static_cast<int64_t>(rois_num);
    return gert::TilingContextPara(
        "RoiAlignRotated",
        {{{{1, 8, 8, 8}, {1, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, r}, {6, r}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{r, pooled_h, pooled_w, 8}, {r, pooled_h, pooled_w, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooled_h)),
            gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooled_w)),
            gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
            gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("clockwise", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend910b", 48, 196608);
}

TEST_F(TilingForRoiAlignRotated, roi_align_rotated_tiling_rois_num_1)
{
    auto para = MakeRoiAlignRotatedTilingPara(1, 2, 2);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(TilingForRoiAlignRotated, roi_align_rotated_tiling_rois_num_7)
{
    auto para = MakeRoiAlignRotatedTilingPara(7, 2, 2);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(TilingForRoiAlignRotated, roi_align_rotated_tiling_rois_num_9)
{
    auto para = MakeRoiAlignRotatedTilingPara(9, 2, 2);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.blockNum, 2U);
}

TEST_F(TilingForRoiAlignRotated, roi_align_rotated_tiling_rois_num_49153)
{
    auto para = MakeRoiAlignRotatedTilingPara(49153, 2, 2);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.blockNum, 48U);
}
