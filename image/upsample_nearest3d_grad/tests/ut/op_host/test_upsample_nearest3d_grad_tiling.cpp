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

#include "../../../op_host/upsample_nearest3d_grad_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace optiling;

class UpsampleNearest3dGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleNearest3dGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleNearest3dGradTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleNearest3dGradTiling, upsample_nearest3d_grad_tiling_001)
{
    UpsampleNearest3dGradCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3dGrad", {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 128, 128, 128})),
         gert::TilingContextPara::OpAttr(
             "output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0, 0.0}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2 1 128 128 128 128 128 128 4575657222473777152 5360320512 64 129 129 4416 0 0 0 0 0 0 0 2 0 0 0 0 0 24 0 0 "
        "16384 0 0 683 0 0 48 70368744177665 549755814016 70368744177792 296352743488 274877907200 8589934624 "
        "4294967299 1 0 387028092977152 65536 4294967297 4294967297 12884901889 0 8589934594 1 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17625088};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dGradTiling, upsample_nearest3d_grad_tiling_002)
{
    UpsampleNearest3dGradCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3dGrad", {{{{1, 1, 64, 64, 64}, {1, 1, 64, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 32, 32, 32}, {1, 1, 32, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 32, 32, 32})),
         gert::TilingContextPara::OpAttr(
             "output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64, 64})),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5, 0.5}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2 1 32 32 32 64 64 64 4539628425446424576 282579840270336 64 129 97 2368 131072 65536 0 0 0 1 1 1 0 0 0 16 1 "
        "48 1024 64 4096 64 64 86 16 1 48 17592186044417 274877906976 17592186044480 158913790016 274877907200 "
        "8589934624 4294967298 1 0 351843720888320 65536 4294967297 4294967297 8589934593 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 137438953473 274877906976 274877907008 158913789984 137438953536 4294967336 4294967297 1 0 65970697666560 "
        "8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 137438953473 274877907968 274877907008 "
        "158913790976 1099511627840 8589934624 4294967298 1 4294967296 351843720888320 65536 4294967297 4294967297 "
        "4294967298 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {18018304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dGradTiling, upsample_nearest3d_grad_tiling_003)
{
    UpsampleNearest3dGradCompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3dGrad", {{{{1, 1, 64, 64, 64}, {1, 1, 64, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 32, 32, 32}, {1, 1, 32, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 32, 32, 32})),
         gert::TilingContextPara::OpAttr(
             "output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64, 64})),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5, 0.5}))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 259;
    string expectTilingData = "512 1 1 64 64 64 32 32 32 0 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
