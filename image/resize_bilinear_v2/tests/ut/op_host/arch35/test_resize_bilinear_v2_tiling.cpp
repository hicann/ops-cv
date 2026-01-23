/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_resize_bilinear_v2_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/resize_bilinear_v2_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ResizeBilinearV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeBilinearV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeBilinearV2Tiling TearDown" << std::endl;
    }
};

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case01)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{2, 256, 12, 12}, {2, 256, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 18, 18}, {2, 256, 18, 18}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5, 1.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData =
        "30001 64 245760 0 0 2 256 12 12 18 18 2 18 18 256 324 2 18 18 256 2592 0 4551638025887197867 324 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case02)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{2, 256, 128, 128}, {2, 256, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 192, 192}, {2, 256, 192, 192}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5, 1.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData =
        "30001 64 245760 1 0 2 256 128 128 192 192 2 192 192 256 36864 2 192 192 256 294912 0 4551512269244741707 "
        "36864 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case03)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{2, 128, 128, 256}, {2, 128, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 64, 256}, {2, 64, 64, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40001;
    string expectTilingData =
        "40001 64 245760 0 0 2 256 128 128 64 64 1 2 64 256 4096 1 1 64 256 0 0 4611686019501129728 4096 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case04)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{2, 128, 128, 256}, {2, 128, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 192, 192, 256}, {2, 192, 192, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5, 1.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData =
        "30000 64 245760 1 0 2 256 128 128 192 192 2 192 192 256 36864 2 192 192 256 294912 0 4551512269244741707 "
        "36864 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case05)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{128, 14, 16, 256}, {128, 14, 16, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{128, 5, 8, 256}, {128, 5, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({3.0, 3.0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData =
        "30000 64 245760 0 1 128 256 14 16 5 8 128 5 8 256 40 128 5 8 256 20480 0 4515609228859845291 40 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case06)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{10, 10, 10, 10}, {10, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{10, 10, 64, 1024}, {10, 10, 64, 1024}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30004;
    string expectTilingData =
        "30004 64 245760 0 0 10 10 10 10 64 1024 10 64 1024 10 65536 10 64 1024 10 1024 0 4539628425446424576 65536 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case07)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 36, 15, 128}, {1, 36, 15, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 12, 5, 128}, {1, 12, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0, 0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40001;
    string expectTilingData =
        "40001 60 245760 0 1 1 128 36 15 12 5 1 1 1 128 60 239 1 1 128 0 0 4629700418014806016 60 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case08)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 36, 15, 128}, {1, 36, 15, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 12, 5, 128}, {1, 12, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0, 0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40001;
    string expectTilingData =
        "40001 60 245760 0 0 1 128 36 15 12 5 1 1 1 128 60 239 1 1 128 0 0 4629700418014806016 60 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case09)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 36, 15, 128}, {1, 36, 15, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 12, 5, 128}, {1, 12, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.2, 0.2})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40001;
    string expectTilingData =
        "40001 60 245760 0 1 1 128 36 15 12 5 1 1 1 128 60 239 1 1 128 0 0 4656722015785320448 60 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case10)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{190, 22, 79, 128}, {190, 22, 79, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{190, 13, 43, 128}, {190, 13, 43, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr(
                "scales",
                Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.31269457412229545, 0.27077088627767687})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData =
        "10000 63 245760 0 1 190 128 22 79 13 43 190 13 43 128 9 95 13 43 128 0 0 4633267427011747001 559 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case11)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 59, 32, 64}, {1, 59, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40003;
    string expectTilingData = "40003 63 245760 1 0 1 64 1 1 59 32 1 59 32 64 30 1 59 32 64 0 0 0 30 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case12)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 11, 21}, {1, 1, 11, 21}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40002;
    string expectTilingData = "40002 58 245760 1 0 1 1 1 1 11 21 1 11 21 1 4 1 11 21 1 0 0 0 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case13)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 11, 21}, {1, 1, 11, 21}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40002;
    string expectTilingData = "40002 58 245760 0 1 1 1 1 1 11 21 1 11 21 1 4 1 11 21 1 0 0 4447918761839889457 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case14)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{12768, 10, 128, 256}, {12768, 10, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{12768, 10, 128, 1024}, {12768, 10, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30005;
    string expectTilingData =
        "30005 64 245760 0 0 12768 10 128 256 128 1024 12768 128 1024 10 131072 12768 128 1024 10 2048 0 "
        "4539628425446424576 131072 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case15)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{12768, 128, 128, 256}, {12768, 128, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{12768, 192, 192, 256}, {12768, 192, 192, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5, 1.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30002;
    string expectTilingData =
        "30002 64 245760 1 0 12768 256 128 128 192 192 12768 192 192 256 36864 12768 192 192 256 1882718208 0 "
        "4551512269244741707 36864 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case16)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{3, 11, 21, 5}, {3, 11, 21, 5}}, ge::DT_BF16, ge::FORMAT_NHWC},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 11, 21, 5}, {3, 11, 21, 5}}, ge::DT_BF16, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 40000;
    string expectTilingData =
        "40000 64 245760 0 0 3 5 11 21 11 21 3 11 21 5 231 3 11 21 61376 54 9 4575657222473777152 231 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_tiling_case17)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 3, 3, 128}, {1, 3, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 5, 5, 128}, {1, 5, 5, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr(
                "scales",
                Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.31269457412229545, 0.27077088627767687})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "10000 50 245760 0 0 1 128 3 3 5 5 1 5 5 64 1 1 5 5 128 0 0 4633267427011747001 25 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_fail_tiling_case01)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0, 0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_fail_tiling_case02)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_fail_tiling_case03)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0, 0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_v2_david_fail_tiling_case04)
{
    optiling::ResizeBilinearV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0, 0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}