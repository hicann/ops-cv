/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../../op_host/arch35/resize_nearest_neighbor_v2_tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeNearestNeighborV2TilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeNearestNeighborV2TilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeNearestNeighborV2TilingTest TearDown" << std::endl;
  }
};

TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_01)
{
    gert::StorageShape inputXShape = {{1, 64, 3, 32}, {1, 64, 3, 32}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 64, 6, 64}, {1, 64, 6, 64}};
    int size_value[2] = {6, 64};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false))
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    uint64_t expectTilingKey = 4105;
    string expectTilingData = "64 1536 0 0 1 64 3 32 6 64 0 0 1 0 64 6144 24576 2048 4096 1 1 1 1 0  0 0 0 0 0 65 0 0 99840 24960 0 4539628425446424576 ";
    std::vector<size_t> expectWorkspaces = {16777216};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_02)
{
    gert::StorageShape inputXShape = {{1, 64, 64, 144}, {1, 64, 64, 144}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 64, 64, 128}, {1, 64, 128, 288}};
    int size_value[2] = {128, 288};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false))
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    uint64_t expectTilingKey = 4106;
    string expectTilingData = "64 71424 0 0 1 64 64 144 128 288 0 0 1 0 64 589824 2359296 9216 18432 1 0 0 4 0 0 0 0 0 0 62 0 3 71424 19008 0 4539628425446424576 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
