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

// Original test 01: NCHW GATHER_ALL_HW
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
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    uint64_t expectTilingKey = 4105;
    string expectTilingData = "64 1536 0 0 1 64 3 32 6 64 0 0 1 0 64 6144 24576 2048 4096 1 1 1 1 0 0 0 0 0 0 67 0 0 102912 25728 0 4539628425446424576 ";
    std::vector<size_t> expectWorkspaces = {16777216};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Original test 02: NCHW GATHER_CUT_H
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_02)
{
    gert::StorageShape inputXShape = {{1, 64, 64, 144}, {1, 64, 64, 144}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 64, 128, 288}, {1, 64, 128, 288}};
    int size_value[2] = {128, 288};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    uint64_t expectTilingKey = 4106;
    string expectTilingData = "64 73728 0 0 1 64 64 144 128 288 0 0 1 0 64 589824 2359296 9216 18432 1 0 0 64 0 0 0 0 0 0 64 0 2 73728 19584 0 4539628425446424576 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 03: NHWC UB2UB W_OUT_ALL strategy
// NHWC {64,4,32,48} -> {64,4,64,48}, integer 2x upscale, N=64 prevents NH core split
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_03)
{
    gert::StorageShape inputXShape = {{64, 4, 32, 48}, {64, 4, 32, 48}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{64, 4, 64, 48}, {64, 4, 64, 48}};
    int size_value[2] = {4, 64};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 04: NHWC UB2UB CORE_NH strategy
// NHWC {2,8,32,64} -> {2,16,64,64}, small N+H triggers CORE_NH
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_04)
{
    gert::StorageShape inputXShape = {{2, 8, 32, 64}, {2, 8, 32, 64}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{2, 16, 64, 64}, {2, 16, 64, 64}};
    int size_value[2] = {16, 64};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 05: NHWC SmallC strategy
// NHWC {1,8,32,64} -> {1,16,75,64}, non-integer ratio, moderate C
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_05)
{
    gert::StorageShape inputXShape = {{1, 8, 32, 64}, {1, 8, 32, 64}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 16, 75, 64}, {1, 16, 75, 64}};
    int size_value[2] = {16, 75};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: NHWC JHC strategy
// NHWC {1,4,8,128} -> {1,12,20,128}, C=128 aligned, small N triggers JHC
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_06)
{
    gert::StorageShape inputXShape = {{1, 4, 8, 128}, {1, 4, 8, 128}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 12, 20, 128}, {1, 12, 20, 128}};
    int size_value[2] = {12, 20};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07: NHWC BigC strategy
// NHWC {1,2,2,128} -> {1,4,5,128}, large C relative to ub
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_07)
{
    gert::StorageShape inputXShape = {{1, 2, 2, 128}, {1, 2, 2, 128}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 4, 5, 128}, {1, 4, 5, 128}};
    int size_value[2] = {4, 5};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 08: NCHW same-size (SIMT_INPUT_EQ_OUTPUT)
// NCHW {1,16,32,32} -> {1,16,32,32}
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_08)
{
    gert::StorageShape inputXShape = {{1, 16, 32, 32}, {1, 16, 32, 32}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 16, 32, 32}, {1, 16, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 09: NCHW downsampling with align_corners=true (SIMT_COMMON)
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_09)
{
    gert::StorageShape inputXShape = {{1, 16, 64, 64}, {1, 16, 64, 64}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 16, 32, 32}, {1, 16, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 10: NCHW downsampling with half_pixel_centers=true (SIMT_COMMON)
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_10)
{
    gert::StorageShape inputXShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 4, 16, 16}, {1, 4, 16, 16}};
    int size_value[2] = {16, 16};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 11: NHWC SmallC strategy (C=63 non-aligned to force isAlign=false)
// isAlign=false skips JHC, falls to SmallC
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_11)
{
    gert::StorageShape inputXShape = {{1, 8, 32, 63}, {1, 8, 32, 63}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 16, 75, 63}, {1, 16, 75, 63}};
    int size_value[2] = {16, 75};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 12: NCHW downsample with explicit scales (0.5, 0.5)
// Coverage: SetScales path with originalScale > 0
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_12)
{
    gert::StorageShape inputXShape = {{1, 64, 64, 64}, {1, 64, 64, 64}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 64, 32, 32}, {1, 64, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 13: NCHW upsampling half_pixel_centers=true
// Coverage: half_pixel branch in CalcSrcLenByDesLen via ComputeGatherCutH
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_13)
{
    gert::StorageShape inputXShape = {{1, 64, 40, 60}, {1, 64, 40, 60}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 64, 80, 120}, {1, 64, 80, 120}};
    int size_value[2] = {80, 120};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 14: NHWC downsample (not-enlarged rejects UB2UB, hits NHWC path)
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_14)
{
    gert::StorageShape inputXShape = {{1, 16, 64, 64}, {1, 16, 64, 64}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{1, 8, 32, 64}, {1, 8, 32, 64}};
    int size_value[2] = {8, 32};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 15: NHWC UB2UB NOT_W_OUT_ALL
// N=30 < 64, H=100 with hCoreNum=50 > realCoreNum=30 triggers condition_=1 → ComputeHCut
TEST_F(ResizeNearestNeighborV2TilingTest, resize_nearest_neighbor_v2_tiling_15)
{
    gert::StorageShape inputXShape = {{30, 100, 64, 128}, {30, 100, 64, 128}};
    gert::StorageShape inputSizeShape = {{2,}, {2,}};
    gert::StorageShape outputShape = {{30, 200, 128, 128}, {30, 200, 128, 128}};
    int size_value[2] = {200, 128};

    ResizeNearestNeighborV2CompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
