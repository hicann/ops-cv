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
 * \file test_resize_bilinear_v2_grad_tiling.cpp
 * \brief test_resize_bilinear_v2_grad_tiling
 */

#include "../../../../op_host/arch35/resize_bilinear_v2_grad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ResizeBilinearV2GradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeBilinearV2GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeBilinearV2GradTiling TearDown" << std::endl;
    }
};

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case01)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 200, 2, 3}, {2, 200, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData =
        "10000 245760 0 1 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 0 64 218 48 64 37 32 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case02)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10002;
    string expectTilingData =
        "10002 245760 1 0 2 1 2 2 4 4 2 4 4 1 16 2 4 4 1 16 4515609228859845291 4629700418014806016 0 0 0 8 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case03)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData =
        "10000 245760 0 1 2 1 4 4 2 2 2 2 2 1 4 2 2 2 1 4 4611686019501129728 0 32 1 0 8 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case04)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10002;
    string expectTilingData =
        "10002 245760 0 0 2 1 4 4 2 2 2 2 2 1 4 2 2 2 1 4 4611686019501129728 4539628425446424576 0 0 0 32 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case05)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 200}, {2, 1, 2, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 1, 4, 200}, {2, 1, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 1, 4, 200}, {2, 1, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10013;
    string expectTilingData =
        "10013 245760 0 0 2 200 1 4 1 2 2 1 2 200 2 2 1 2 200 2 4611686019501129728 4539628425446424576 0 0 0 64 25 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case06)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string expectTilingData =
        "20000 245760 0 1 2 200 5 7 2 3 1 2 3 40 1 1 2 3 200 6 4720507691772089740 0 64 218 48 60 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case07)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10013;
    string expectTilingData =
        "10013 245760 0 0 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 4430100883889375413 0 0 0 64 218 48 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case08)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 200, 2, 3}, {2, 200, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10012;
    string expectTilingData =
        "10012 245760 0 0 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 4430100883889375413 0 0 0 64 218 48 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case09)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 6, 9, 200}, {2, 6, 9, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 6, 9, 200}, {2, 6, 9, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData =
        "30001 245760 1 0 2 200 6 9 2 3 1 1 1 40 6 767 1 1 40 6 4647714816530579456 0 64 337 32 60 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_david_tiling_case10)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData =
        "30000 245760 1 0 2 200 2 3 2 3 2 2 3 200 6 2 2 3 30688 6 4575657222473777152 0 0 0 0 64 37 32 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}