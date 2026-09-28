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
#include <stdint.h>
#include <iostream>
#include <limits>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_kernel/arch35/image_projective_transform_tiling_data.h"
#include "../../../../op_kernel/arch35/image_projective_transform_tiling_key.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ImageProjectiveTransformTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ImageProjectiveTransformTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ImageProjectiveTransformTiling TearDown" << std::endl; }
};

struct ImageProjectiveTransformCompileInfo {};

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float32_bilinear)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "1 16 17179869185 17179869188 12884901892 16 ", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float32_nearest)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("NEAREST")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "1 16 17179869185 17179869188 12884901892 16 ", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float16_bilinear)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "1 16 17179869185 17179869188 12884901892 16 ", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_uint8_bilinear)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_UINT8, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "1 16 17179869185 17179869188 12884901892 16 ", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_int32_nearest)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_INT32, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_INT32, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("NEAREST")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "1 16 17179869185 17179869188 12884901892 16 ", expectWorkspaces);
}

namespace {

gert::TilingContextPara MakeOutputShapeContractCase(const std::vector<int32_t>& outputShapeData,
                                                    const std::vector<int64_t>& outputShape)
{
    static ImageProjectiveTransformCompileInfo compileInfo = {};
    const int64_t N = 1;
    const int64_t HIn = 4;
    const int64_t WIn = 4;
    const int64_t C = 3;
    return gert::TilingContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(outputShapeData.data())}},
        {{{{outputShape[0], outputShape[1], outputShape[2], outputShape[3]},
           {outputShape[0], outputShape[1], outputShape[2], outputShape[3]}},
          ge::DT_FLOAT,
          ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
}

gert::TilingContextPara MakeLargeOutputShapeProbe(std::vector<int32_t>& outputShapeData, int64_t batchSize)
{
    static ImageProjectiveTransformCompileInfo compileInfo = {};
    constexpr int64_t hIn = 4;
    constexpr int64_t wIn = 4;
    constexpr int64_t channels = 3;
    return gert::TilingContextPara(
        "ImageProjectiveTransform",
        {{{{batchSize, hIn, wIn, channels}, {batchSize, hIn, wIn, channels}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{batchSize, 8}, {batchSize, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{batchSize, outputShapeData[0], outputShapeData[1], channels},
           {batchSize, outputShapeData[0], outputShapeData[1], channels}},
          ge::DT_FLOAT,
          ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
        &compileInfo, "Ascend950");
}

void PrintLargeOutputShapeTiling(const char* label, const TilingInfo& info)
{
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    std::cout << label << " status=GRAPH_SUCCESS block_dim=" << info.blockNum << " batch=" << data->batchSize
              << " h_out=" << data->hOut << " w_out=" << data->wOut << " total_pixels=" << data->totalPixels
              << " spatial_size=" << data->spatialSize << std::endl;
}

} // namespace

TEST_F(ImageProjectiveTransformTiling, output_shape_zero_preserves_empty_output)
{
    std::vector<int32_t> outputShapeData = {0, 3};
    auto para = MakeOutputShapeContractCase(outputShapeData, {1, 0, 3, 3});
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.blockNum, 1);
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->hOut, 0);
    EXPECT_EQ(data->wOut, 3);
    EXPECT_EQ(data->totalPixels, 0);
    EXPECT_EQ(data->spatialSize, 0);
}

TEST_F(ImageProjectiveTransformTiling, output_shape_width_zero_preserves_empty_output)
{
    std::vector<int32_t> outputShapeData = {2, 0};
    auto para = MakeOutputShapeContractCase(outputShapeData, {1, 2, 0, 3});
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.blockNum, 1);
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->hOut, 2);
    EXPECT_EQ(data->wOut, 0);
    EXPECT_EQ(data->totalPixels, 0);
    EXPECT_EQ(data->spatialSize, 0);
}

TEST_F(ImageProjectiveTransformTiling, output_shape_negative_is_rejected)
{
    std::vector<int32_t> outputShapeData = {-1, 3};
    auto para = MakeOutputShapeContractCase(outputShapeData, {1, -1, 3, 3});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ImageProjectiveTransformTiling, output_shape_int32_max_single_dimension_is_positive_control)
{
    constexpr int64_t batchSize = 3;
    constexpr int32_t maxDim = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> outputShapeData = {maxDim, 1};
    auto context = MakeLargeOutputShapeProbe(outputShapeData, batchSize);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    PrintLargeOutputShapeTiling("output_shape=[INT32_MAX,1]", info);
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->hOut, maxDim);
    EXPECT_EQ(data->wOut, 1);
    EXPECT_EQ(data->totalPixels, static_cast<int64_t>(batchSize) * maxDim);
    EXPECT_GT(info.blockNum, 0U);
}

TEST_F(ImageProjectiveTransformTiling, output_shape_int32_max_product_is_rejected)
{
    constexpr int64_t batchSize = 3;
    constexpr int32_t maxDim = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> outputShapeData = {maxDim, maxDim};
    auto context = MakeLargeOutputShapeProbe(outputShapeData, batchSize);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST_F(ImageProjectiveTransformTiling, output_shape_int32_max_product_with_two_batches_fits_int64)
{
    constexpr int64_t batchSize = 2;
    constexpr int32_t maxDim = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> outputShapeData = {maxDim, maxDim};
    auto context = MakeLargeOutputShapeProbe(outputShapeData, batchSize);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    PrintLargeOutputShapeTiling("output_shape=[INT32_MAX,INT32_MAX],batch=2", info);
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    const int64_t expectedTotal = static_cast<int64_t>(batchSize) * maxDim * maxDim;
    EXPECT_EQ(data->totalPixels, expectedTotal);
    EXPECT_GT(data->totalPixels, 0);
    EXPECT_GT(info.blockNum, 0U);
}
