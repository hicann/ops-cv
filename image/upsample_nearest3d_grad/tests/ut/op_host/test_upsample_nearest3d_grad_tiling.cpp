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