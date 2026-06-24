/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/resize_nearest_neighbor_v2_grad_tiling_base.h"

class ResizeNearestNeighborV2GradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeNearestNeighborV2GradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeNearestNeighborV2GradTiling TearDown" << std::endl; }
};

struct ResizeNearestNeighborV2GradCompileInfo {
    int32_t core_num = 0;
    int32_t ubSize = 0;
};

// Test 01: NCHW same size (ALL_COPY)
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_nchw_fp32)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {2, 2};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 02: NCHW upsample FP16 (DETERMINE)
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_nchw_fp16)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {4, 4};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 03: NCHW upsample BF16 (DETERMINE)
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_nchw_bf16)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {8, 8};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::DT_BF16, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{2, 3, 8, 8}, {2, 3, 8, 8}}, ge::DT_BF16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 04: NHWC upsample FP32 (DETERMINE with NHWC)
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_nhwc_fp32)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {4, 4};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 2, 2, 1}, {1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 05: NCHW large upsample FP32 (DETERMINE)
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_large_nchw_fp32)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{2, 3, 50, 50}, {2, 3, 50, 50}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{2, 3, 100, 100}, {2, 3, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: NOT_DETERMINE path with align_corners=true
// Coverage: DoTilingSimtNotDetermine, SetScales align_corners+!isDetermine
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_not_determine_align_corners)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{2, 3, 60, 60}, {2, 3, 60, 60}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{2, 3, 100, 100}, {2, 3, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07: NOT_DETERMINE_HW path with align_corners and large HW ratio
// Coverage: DoTilingSimtNotDetermineHW, IsSimtNotDetermineHW
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_not_determine_hw)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 200};
    // dstW=2400 >= 2048, ratio ~12, srcHW=20000 >= 4096
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 1200, 2400}, {1, 1, 1200, 2400}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 100, 200}, {1, 1, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 08: DETERMINE_HW path
// Coverage: DoTilingSimtDetermineHW, IsSimtDetermineHW
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_determine_hw)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 200};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 1200, 2400}, {1, 1, 1200, 2400}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 100, 200}, {1, 1, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 09: DETERMINE_1D path (W=1, H mismatched)
// Coverage: DETERMINE_1D tiling key, FillTilingData H/W swap branch
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_determine_1d)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 1};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 3, 200, 1}, {1, 3, 200, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 3, 100, 1}, {1, 3, 100, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 10: DETERMINE with half_pixel_centers=true
// Coverage: halfPixelCenters=TRUE in tiling key
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_half_pixel_determine)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {1000, 1000};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 500, 500}, {1, 1, 500, 500}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 1000, 1000}, {1, 1, 1000, 1000}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 11: Error case - align_corners && half_pixel_centers both true
// Coverage: CheckAttrsParams error return
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_error_align_and_halfpixel)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 100, 100}, {1, 1, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 100, 100}, {1, 1, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 12: Error case - N-dim mismatch
// Coverage: SetDims N-dim error
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_error_n_dim_mismatch)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{2, 3, 100, 100}, {2, 3, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 3, 100, 100}, {1, 3, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 13: Error case - C-dim mismatch (NCHW)
// Coverage: SetDims NCHW C-dim error
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_error_c_dim_mismatch_nchw)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 4, 100, 100}, {1, 4, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 3, 100, 100}, {1, 3, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 14: DETERMINE_HW with half_pixel=true
// Coverage: DETERMINE_HW strategy with halfPixelCenters=TRUE tiling key
TEST_F(ResizeNearestNeighborV2GradTiling, resize_nearest_neighbor_v2_grad_tiling_determine_hw_halfpixel)
{
    ResizeNearestNeighborV2GradCompileInfo compileInfo = {1, 253952};
    int32_t sizeData[] = {100, 200};
    gert::TilingContextPara tilingContextPara(
        "ResizeNearestNeighborV2Grad",
        {{{{1, 1, 1200, 2400}, {1, 1, 1200, 2400}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         gert::TilingContextPara::TensorDescription(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    sizeData)},
        {{{{1, 1, 100, 200}, {1, 1, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0, 0.0}))},
        &compileInfo, "Ascend950", 1, 253952);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
