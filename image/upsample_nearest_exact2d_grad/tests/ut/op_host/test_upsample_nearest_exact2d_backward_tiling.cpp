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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_nearest_exact2d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class UpsampleNearestExact2dGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleNearestExact2dGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleNearestExact2dGradTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleNearestExact2dGradTiling, upsample_upsample_nearest_exact2d_grad_tiling_001)
{
    optiling::UpsampleNearestExact2dGradCompileInfo compileInfo = {1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearestExact2dGrad",
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})),
         gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 128})),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(1)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(1))},
        &compileInfo, socVersion, 20, 192 * 1024, 10240);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "64 2 4575657222473777152 4575657222473777152 0 0 512 274877911296 1 1 1 128 1 1 1 128 4294967296 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 4294967297 549755814016 274877907072 4294967424 549755813952 4294967304 4294967297 1 0 "
        "26388279066624 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33593344};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearestExact2dGradTiling, upsample_upsample_nearest_exact2d_grad_tiling_002)
{
    optiling::UpsampleNearestExact2dGradCompileInfo compileInfo = {1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearestExact2dGrad",
        {
            {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 64})),
         gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 32})),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(2)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(2))},
        &compileInfo, socVersion, 20, 192 * 1024, 10240);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "64 1 4611686019501129728 4539628425446424576 0 0 128 549755817984 1 1 2 64 1 1 1 32 4294967297 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8589934593 274877906976 "
        "8589934656 274877907008 274877906960 4294967360 4294967297 1 0 43980465111040 4096 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967297 274877906976 274877907008 8589934624 137438953536 "
        "4294967312 4294967297 1 0 13194139533312 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {33588224};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearestExact2dGradTiling, upsample_upsample_nearest_exact2d_grad_tiling_003)
{
    optiling::UpsampleNearestExact2dGradCompileInfo compileInfo = {1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearestExact2dGrad",
        {
            {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 32}, {1, 1, 1, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 64})),
         gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 32})),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(2)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(2))},
        &compileInfo, socVersion, 20, 192 * 1024, 10240);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "64 3 4611686019501129728 4539628425446424576 0 0 128 549755817984 1 1 2 64 1 1 1 32 4294967297 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8589934593 274877906976 "
        "8589934656 274877907008 274877906960 4294967360 4294967297 1 0 43980465111040 4096 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967297 274877906976 274877907008 8589934624 137438953536 "
        "4294967312 4294967297 1 0 13194139533312 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {33588224};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearestExact2dGradTiling, upsample_upsample_nearest_exact2d_grad_tiling_004)
{
    optiling::UpsampleNearestExact2dGradCompileInfo compileInfo = {1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest2dGrad",
        {
            {{{1, 1, 1, 4, 16}, {1, 1, 1, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            {{{1, 1, 1, 4, 16}, {1, 1, 1, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 4})),
         gert::TilingContextPara::OpAttr(
             "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 4})),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(1)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(1))},
        &compileInfo, socVersion, 40, 192 * 1024, 10240);
    uint64_t expectTilingKey = 101;
    string expectTilingData =
        "16000 1 127 4575657222473777152 0 4 1 16842752 1 1 1 4 1 1 1 4 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    std::vector<size_t> expectWorkspaces = {1048576};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}