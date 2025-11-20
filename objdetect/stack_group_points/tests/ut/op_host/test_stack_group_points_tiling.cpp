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
#include "../../../op_host/stack_group_points_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;

class StackGroupPointsTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "test TestStackGroupPointsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "test TestStackGroupPointsTiling TearDown" << std::endl;
    }
};

TEST_F(StackGroupPointsTiling, fp32)
{
    optiling::StackGroupPointsCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                                {{{{5, 12}, {5, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{10, 6}, {10, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{10, 12, 6}, {10, 12, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);

    uint64_t expectTilingKey = 1;
    string expectTilingData = "2 3 12 -4702111234474983746 -4238682002231055 256 32 32 32 -4 5425512962855750368 5 -3255307777713450312 48 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}