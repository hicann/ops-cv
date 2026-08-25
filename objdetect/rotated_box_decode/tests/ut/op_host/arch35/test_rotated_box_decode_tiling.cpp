/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/rotated_box_decode_tiling_data.h"

class RotatedBoxDecodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RotatedBoxDecodeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RotatedBoxDecodeTiling TearDown" << std::endl; }
};

struct RotatedBoxDecodeCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

using TTD = gert::TilingContextPara::TensorDescription;

static TilingInfo RunTiling(ge::DataType dtype, int64_t B, int64_t N, const std::vector<float>& weight)
{
    RotatedBoxDecodeCompileInfo compileInfo;
    gert::StorageShape boxS({B, 5, N}, {B, 5, N});
    gert::TilingContextPara tilingContextPara(
        "RotatedBoxDecode", {TTD(boxS, dtype, ge::FORMAT_ND), TTD(boxS, dtype, ge::FORMAT_ND)},
        {TTD(boxS, dtype, ge::FORMAT_ND)},
        {gert::TilingContextPara::OpAttr("weight", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>(weight))},
        &compileInfo, "Ascend950", 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    return tilingInfo;
}

// KEY=0 (N-axis multicore, large N): shape (2, 5, 65473), weight default
TEST_F(RotatedBoxDecodeTiling, rotated_box_decode_tiling_key_n_axis)
{
    auto info = RunTiling(ge::DT_FLOAT, 2, 65473, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_EQ(info.tilingKey, 0);
    auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, 0);
    EXPECT_EQ(td->B, 2);
    EXPECT_EQ(td->N, 65473);
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->copyMode, 0);
    EXPECT_EQ(td->totalCount, 2 * 65473);
}

// KEY=1 (B-axis fullload, small N large B): shape (100, 5, 8), weight default
TEST_F(RotatedBoxDecodeTiling, rotated_box_decode_tiling_key_b_axis)
{
    auto info = RunTiling(ge::DT_FLOAT, 100, 8, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_EQ(info.tilingKey, 256);
    auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, 1);
    EXPECT_EQ(td->B, 100);
    EXPECT_EQ(td->N, 8);
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->copyMode, 0);
    EXPECT_EQ(td->totalCount, 100 * 8);
}
