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
#include "../../../op_host/upsample_linear1d_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class UpsampleLinear1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleLinear1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleLinear1dTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleLinear1dTiling, upsample_linear1d_tiling_001)
{
    optiling::UpsampleLinear1dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleLinear1d",
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "1 0 64 16 1 4575657222482165760 0 8192 0 0 0 0 1 64 0 0 0 0 0 0 0 1 1 1 128 1 1 1 0 4294967297 549755813888 "
        "4294967424 549755813952 274877906960 4294967424 4294967297 1 0 87960930222080 4096 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33570816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}