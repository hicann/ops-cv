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
#include "../../../op_host/grid_sampler3_d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampler3DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSampler3DGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSampler3DGradTiling TearDown" << std::endl;
    }
};

struct GridSampler3DGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 0 206158430209 68719476757 1099511627777 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
