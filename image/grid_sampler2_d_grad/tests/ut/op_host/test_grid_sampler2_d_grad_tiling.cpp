/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "../../../op_host/grid_sampler_2d_grad_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampler2DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSampler2DGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSampler2DGradTiling TearDown" << std::endl;
    }
};

struct GridSampler2DGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{output1, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "8589934594 12884901920 68719476752 34359738376 3573412790320 0 4294967297 206158430209 32 28874 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
