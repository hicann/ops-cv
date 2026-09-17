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
#include <iostream>

#include <gtest/gtest.h>

#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_data.h"
#include "tiling_case_executor.h"

namespace {

std::vector<gert::TilingContextPara::OpAttr> RoiAlignAttrs(int64_t pooledHeight, int64_t pooledWidth)
{
    return {
        gert::TilingContextPara::OpAttr("pooled_height", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledHeight)),
        gert::TilingContextPara::OpAttr("pooled_width", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledWidth)),
        gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0F)),
        gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
    };
}

gert::TilingContextPara MakeParams(int64_t pooledHeight, int64_t pooledWidth)
{
    return gert::TilingContextPara(
        "RoiAlignV2",
        {{{{1, 3, 4, 4}, {1, 3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 3, 2, 2}, {1, 3, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, RoiAlignAttrs(pooledHeight, pooledWidth));
}

} // namespace

TEST(RoiAlignV2PooledSizeInt32NarrowingTiling, valid_control_succeeds)
{
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(2, 2), info);
    std::cout << "case=valid_control tiling_success=" << ok << " expected=true\n";
    EXPECT_TRUE(ok);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingTiling, int32_max_plus_one_is_rejected_after_narrowing)
{
    const int64_t outOfRange = static_cast<int64_t>(INT32_MAX) + 1;
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(outOfRange, 2), info);
    std::cout << "case=height_above_int32 tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingTiling, width_int32_max_plus_one_is_rejected_before_narrowing)
{
    const int64_t outOfRange = static_cast<int64_t>(INT32_MAX) + 1;
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(2, outOfRange), info);
    std::cout << "case=width_above_int32 tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingTiling, rejects_height_above_uint32)
{
    const int64_t outOfRange = static_cast<int64_t>(UINT32_MAX) + 2;
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(outOfRange, 2), info);
    std::cout << "case=height_above_uint32 tiling_success=" << ok << " expected=false\n";
    if (ok) {
        const auto* data = reinterpret_cast<const RoiAlignV2TilingData*>(info.tilingData.get());
        ASSERT_NE(data, nullptr);
        std::cout << "case=height_above_uint32 actual_pooled_height=" << data->pooledHeight
                  << " actual_pooled_width=" << data->pooledWidth << "\n";
    }
    EXPECT_FALSE(ok);
}

TEST(RoiAlignV2PooledSizeInt32NarrowingTiling, rejects_width_above_uint32)
{
    const int64_t outOfRange = static_cast<int64_t>(UINT32_MAX) + 2;
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(2, outOfRange), info);
    std::cout << "case=width_above_uint32 tiling_success=" << ok << " expected=false\n";
    if (ok) {
        const auto* data = reinterpret_cast<const RoiAlignV2TilingData*>(info.tilingData.get());
        ASSERT_NE(data, nullptr);
        std::cout << "case=width_above_uint32 actual_pooled_height=" << data->pooledHeight
                  << " actual_pooled_width=" << data->pooledWidth << "\n";
    }
    EXPECT_FALSE(ok);
}
