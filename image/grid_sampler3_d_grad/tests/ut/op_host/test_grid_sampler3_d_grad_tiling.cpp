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
    bool regBase{false};
};

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
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
    string expectTilingData =
        "12884901890 68719476752 34359738384 34359738376 0 206158430209 68719476757 1099511627777 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case2)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "12884901890 68719476752 34359738384 34359738376 8589934592 206158430209 68719476757 1099511627777 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case3)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case4)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 300}, {2, 8, 8, 8, 300}};
    gert::StorageShape x = {{2, 16, 16, 16, 300}, {2, 300, 16, 16, 300}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 300}, {2, 300, 16, 16, 300}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "1288490188802 68719477036 34359738384 34359738376 4294967297 206158430209 68719476757 5222680231952 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case5)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 600}, {2, 8, 8, 8, 600}};
    gert::StorageShape x = {{2, 16, 16, 16, 600}, {2, 16, 16, 16, 600}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 600}, {2, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2576980377602 68719476752 34359738384 34359738376 1 206158430209 68719476757 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case6)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case7)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case8)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("xxx")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case9)
{
    gert::StorageShape grad = {{1, 3, 3, 3, 3}, {1, 3, 3, 3, 3}};
    gert::StorageShape x = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "12884901889 68719476752 8589934608 8589934594 1 34359738369 1 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, 0, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case10)
{
    gert::StorageShape grad = {{1, 3, 3, 3, 3}, {1, 3, 3, 3, 3}};
    gert::StorageShape x = {{1, 16, 16, 3}, {1, 16, 16, 3}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "12884901889 68719476752 8589934595 8589934594 1 34359738369 1 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, 0, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case11)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 1600}, {2, 8, 8, 8, 1600}};
    gert::StorageShape x = {{2, 16, 16, 16, 1600}, {2, 16, 16, 16, 1600}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 1600}, {2, 16, 16, 16, 1600}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "6871947673602 68719476752 34359738384 34359738376 1 206158430209 68719476757 5772436045825 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case12)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "68719476738 68719476752 34359738371 34359738376 1 51539607553 0 6322191859744 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {x, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_bfloat16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_BF16, ge::FORMAT_ND},
                                                {x, ge::DT_BF16, ge::FORMAT_ND},
                                                {grid, ge::DT_BF16, ge::FORMAT_ND}},
                                                {{dx, ge::DT_BF16, ge::FORMAT_ND},
                                                {dgrid, ge::DT_BF16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
