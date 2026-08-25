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

#include <gtest/gtest.h>

#include "any_value.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/arch35/non_max_suppression_v7_tiling_data.h"

namespace {
struct CompileInfo {
    uint32_t coreNum{64};
    uint64_t ubSize{262144};
};

gert::TilingContextPara MakeContext(int64_t boxes, int64_t classes, int64_t maxBoxes, uint64_t ubSize = 262144)
{
    static CompileInfo compileInfo;
    compileInfo = CompileInfo{64, ubSize};
    return gert::TilingContextPara(
        "NonMaxSuppressionV7",
        {
            {gert::StorageShape({1, boxes, 4}, {1, boxes, 4}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({1, classes, boxes}, {1, classes, boxes}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND},
            {gert::StorageShape({1}, {1}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({1}, {1}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({1, classes, boxes, 3}, {1, classes, boxes, 3}), ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {gert::StorageShape({maxBoxes, 3}, {maxBoxes, 3}), ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("center_point_box", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("max_boxes_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxBoxes)),
        },
        &compileInfo, "Ascend950", 64, ubSize);
}

const NonMaxSuppressionV7TilingData* GetTilingData(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingDataSize, sizeof(NonMaxSuppressionV7TilingData));
    return reinterpret_cast<const NonMaxSuppressionV7TilingData*>(info.tilingData.get());
}
} // namespace

TEST(NonMaxSuppressionV7Tiling, empty_output_does_not_allocate_vector_scratch)
{
    auto context = MakeContext(1000000, 80, 0);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->scratchFieldStride, 0U);
    EXPECT_EQ(tiling->classIndicesOffset, 0U);
    EXPECT_EQ(tiling->classCountsOffset, 0U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_LT(info.workspaceSizes[0], 128U * 1024U * 1024U);
}

TEST(NonMaxSuppressionV7Tiling, nonempty_serial_path_only_reserves_selected_boxes)
{
    auto context = MakeContext(16, 80, 4);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->scratchFieldStride, 0U);
    EXPECT_EQ(tiling->classIndicesOffset, 0U);
    EXPECT_EQ(tiling->classCountsOffset, 32U);
}

TEST(NonMaxSuppressionV7Tiling, vector_path_keeps_vector_scratch)
{
    auto context = MakeContext(17, 2, 64);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_GT(tiling->scratchFieldStride, 0U);
    EXPECT_GT(tiling->classIndicesOffset, 0U);
}

TEST(NonMaxSuppressionV7Tiling, vector_tile_excludes_simt_data_cache)
{
    auto context = MakeContext(1000, 2, 64, 64U * 1024U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    // 64 KiB total - 32 KiB DCache - 16 KiB fixed reserve leaves 16 KiB.
    // floor(16 KiB / (11 * sizeof(float) + sizeof(uint8_t))) aligned to 64 is 320.
    EXPECT_EQ(tiling->tileSize, 320);
}
