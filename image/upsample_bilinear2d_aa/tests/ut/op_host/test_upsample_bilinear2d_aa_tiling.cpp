/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bilinear2d_aa_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class UpsampleBilinear2dAATiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBilinear2dAATiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBilinear2dAATiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleBilinear2dAATiling, upsample_bilinear2d_aa_tiling_001)
{
    UpsampleBilinear2dAACompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dAA",
                                              {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
                                                {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
                                                {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
                                                {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 2 4539628425446424576 4575657222473777152 13950255104 13950255104 131072 9620726745280 17179869192 0 0 4 2 64 4 0 0 4 1 1 4 1 1 128 128 1 1 256 256 549755813889 549755814144 549755814016 150323855424 274877907072 4294967336 4294967297 1 0 131941395333120 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 150323855616 1099511627840 8589934624 4294967298 1 0 351843720888320 65536 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33757184};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dAATiling, upsample_bilinear2d_aa_tiling_002)
{
    UpsampleBilinear2dAACompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {128, 128};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dAA",
                                              {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
                                                {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
                                                {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
                                                {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 2 4575657222473777152 4575657222473777152 13950255104 13950255104 65536 18416819765248 8589934592 0 0 0 0 0 0 0 0 2 1 1 2 1 1 128 128 1 1 128 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814016 274877907072 287762808960 549755813952 8589934656 4294967298 1 0 422212465065984 32768 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33654272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
