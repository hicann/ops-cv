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
#include <stdint.h>
#include <iostream>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/grid_sample_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampleTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSampleTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSampleTiling TearDown" << std::endl;
    }
};

struct GridSampleCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isDavid = false;
};

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_1)
{
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 200, 200, 1}, {2, 200, 200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 2, 2, 1}, {2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000220;
    string expectTilingData = "64 2 1 0 200 200 0 2 2 0 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_2)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000220;
    string expectTilingData = "64 2 1 0 200 200 0 2 2 0 1 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_3)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000220;
    string expectTilingData = "64 2 1 0 200 200 0 2 2 0 2 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float16)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000210;
    string expectTilingData = "64 2 1 0 200 200 0 2 2 0 2 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_nearest)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000221;
    string expectTilingData = "64 2 1 0 2 2 0 2 2 1 2 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_bicubic)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000222;
    string expectTilingData = "64 2 1 0 2 2 0 2 2 2 2 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_slicing_window)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1001220;
    string expectTilingData = "64 2 1 0 200 200 0 2 2 0 1 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c2)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 2001220;
    string expectTilingData = "64 2 2 0 2 2 0 2 2 0 1 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c1)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 2101220;
    string expectTilingData = "64 2 1 0 2 2 0 2 2 0 1 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c32)
{
    int64_t N = 2;
    int64_t x_h = 10;
    int64_t x_w = 10;
    int64_t C = 32;
    int64_t grid_h = 10;
    int64_t grid_w = 10;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_h, x_w, C}, {N, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_h, grid_w, dim}, {N, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_h, grid_w, C}, {N, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 2201220;
    string expectTilingData = "64 2 32 0 10 10 0 10 10 0 1 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_1)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_d, x_h, x_w, C}, {N, x_d, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_d, grid_h, grid_w, C}, {N, grid_d, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1010320;
    string expectTilingData = "64 2 1 200 200 200 2 2 2 0 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_2)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_d, x_h, x_w, C}, {N, x_d, x_h, x_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{N, grid_d, grid_h, grid_w, C}, {N, grid_d, grid_h, grid_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1010310;
    string expectTilingData = "64 2 1 200 200 200 2 2 2 0 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_nearest)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_d, x_h, x_w, C}, {N, x_d, x_h, x_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, grid_d, grid_h, grid_w, C}, {N, grid_d, grid_h, grid_w, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1010321;
    string expectTilingData = "64 2 1 200 200 200 2 2 2 1 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_nearest)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, x_d, x_h, x_w, C}, {N, x_d, x_h, x_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{N, grid_d, grid_h, grid_w, C}, {N, grid_d, grid_h, grid_w, C}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1010311;
    string expectTilingData = "64 2 1 200 200 200 2 2 2 1 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_bilinear_te)
{
    int64_t N = 88;
    int64_t x_d = 16;
    int64_t x_h = 64;
    int64_t x_w = 64;
    int64_t C = 4;
    int64_t grid_d = 16;
    int64_t grid_h = 64;
    int64_t grid_w = 64;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, C, x_d, x_h, x_w};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, C, grid_d, grid_h, grid_w};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, C, x_d, x_h, x_w}, {N, C, x_d, x_h, x_w}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{N, C, grid_d, grid_h, grid_w}, {N, C, grid_d, grid_h, grid_w}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1011310;
    string expectTilingData = "64 88 4 16 64 64 16 64 64 0 0 1 0 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17301504};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_bilinear_te)
{
    int64_t N = 22;
    int64_t x_d = 16;
    int64_t x_h = 64;
    int64_t x_w = 64;
    int64_t C = 4;
    int64_t grid_d = 16;
    int64_t grid_h = 64;
    int64_t grid_w = 64;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, C, x_d, x_h, x_w};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, C, grid_d, grid_h, grid_w};
    GridSampleCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{N, C, x_d, x_h, x_w}, {N, C, x_d, x_h, x_w}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{N, grid_d, grid_h, grid_w, dim}, {N, grid_d, grid_h, grid_w, dim}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{N, C, grid_d, grid_h, grid_w}, {N, C, grid_d, grid_h, grid_w}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1011320;
    string expectTilingData = "64 22 4 16 64 64 16 64 64 0 0 1 0 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampleTiling, grid_sample_tiling_david_test_float32_1)
{
    GridSampleCompileInfo compileInfo = {56, 196608, true};
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 1, 200, 200}, {2, 1, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1000;
    string expectTilingData = "64 2 1 200 200 2 2 0 0 1 0 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}