/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_host/resize_upsample_trilinear_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ResizeUpsampleTrilinearArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeUpsampleTrilinearArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeUpsampleTrilinearArch35TilingTest TearDown" << std::endl;
    }
};

struct ResizeUpsampleTrilinearArch35CompileInfo {
    int32_t coreNum = 0;
    int32_t socVersionType = 0;
};

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_output_size_case01)
{
    gert::StorageShape inputShape = {{1, 1, 4, 128, 128}, {1, 1, 4, 128, 128}};
    gert::StorageShape outShape = {{1, 1, 12, 256, 256}, {1, 1, 12, 256, 256}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {12, 256, 256};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "3298534883330 4398046511168 1024 786432 1 4 128 128 12 256 256 4539628425440832171 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp16_output_size_case02)
{
    gert::StorageShape inputShape = {{1, 2, 8, 64, 64}, {1, 2, 8, 64, 64}};
    gert::StorageShape outShape = {{1, 2, 16, 128, 128}, {1, 2, 16, 128, 128}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {16, 128, 128};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "4398046511105 2199023255616 512 524288 2 8 64 64 16 128 128 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_bf16_output_size_case03)
{
    gert::StorageShape inputShape = {{2, 3, 10, 48, 48}, {2, 3, 10, 48, 48}};
    gert::StorageShape outShape = {{2, 3, 20, 96, 96}, {2, 3, 20, 96, 96}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {20, 96, 96};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_BF16, ge::FORMAT_ND}},
        {{outShape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "3092376453123 6597069766720 1536 1105920 6 10 48 48 20 96 96 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_align_corners_case04)
{
    gert::StorageShape inputShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    gert::StorageShape outShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {8, 8, 8};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4294967297 2199023255553 512 512 1 4 4 4 8 8 8 4529334484052176311 5349535159 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_scales_case05)
{
    gert::StorageShape inputShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    gert::StorageShape outShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(2.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(2.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(2.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4294967297 2199023255553 512 512 1 4 4 4 8 8 8 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_small_shape_case06)
{
    gert::StorageShape inputShape = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
    gert::StorageShape outShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {3, 3, 3};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4294967297 137438953473 27 27 1 2 2 2 3 3 3 4551638025887197867 1059760811 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_downsample_case07)
{
    gert::StorageShape inputShape = {{1, 2, 60, 256, 512}, {1, 2, 60, 256, 512}};
    gert::StorageShape outShape = {{1, 2, 30, 128, 256}, {1, 2, 30, 128, 256}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {30, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4123168604164 8796093022272 2048 1966080 2 60 256 512 30 128 256 4611686019501129728 1073741824 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_large_shape_case08)
{
    gert::StorageShape inputShape = {{4, 8, 32, 64, 64}, {4, 8, 32, 64, 64}};
    gert::StorageShape outShape = {{4, 8, 64, 128, 128}, {4, 8, 64, 128, 128}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {64, 128, 128};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4398046511168 140737488355392 32768 33554432 32 32 64 64 64 128 128 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_fp32_scale_exceed_max_case09)
{
    gert::StorageShape inputShape = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
    gert::StorageShape outShape = {{1, 1, 101, 101, 101}, {1, 1, 101, 101, 101}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {101, 101, 101};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearArch35TilingTest, upsample_trilinear3d_nc_mismatch_case10)
{
    gert::StorageShape inputShape = {{1, 2, 4, 4, 4}, {1, 2, 4, 4, 4}};
    gert::StorageShape outShape = {{2, 2, 8, 8, 8}, {2, 2, 8, 8, 8}};
    ResizeUpsampleTrilinearArch35CompileInfo compileInfo = {64, 3510};
    std::vector<int64_t> output_size = {8, 8, 8};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear",
        {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo,
        "Ascend950",
        64,
        262144);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}