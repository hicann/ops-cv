/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

class UpsampleNearestExact2dGradTiling : public testing::Test {
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
        &compileInfo, 20, 192 * 1024, 10240);
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
