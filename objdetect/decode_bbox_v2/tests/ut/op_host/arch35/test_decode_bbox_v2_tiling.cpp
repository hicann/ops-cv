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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/decode_bbox_v2_tiling_struct.h"

using namespace std;
using namespace ge;

class DecodeBboxV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DecodeBboxV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DecodeBboxV2Tiling TearDown" << std::endl; }
};

static std::map<std::string, std::string> soc_versions_infos = {{"Short_SoC_version", "Ascend950"}};

constexpr size_t DBV2_SYS_WORKSPACE_SIZE = 16777216;

TEST_F(DecodeBboxV2Tiling, dbv2_fp32_normal)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 1 512 1 2304 1 512 1 8 4575657222473777152 4575657222473777152 4575657221408423936 "
                    "4575657222473777152 4539628425454813184 ",
                    {0});
}

TEST_F(DecodeBboxV2Tiling, dbv2_fp16_normal)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 1 512 1 1536 1 512 1 8 4575657222473777152 4575657222473777152 4575657221408423936 "
                    "4575657222473777152 4539628425454813184 ",
                    {0});
}

TEST_F(DecodeBboxV2Tiling, dbv2_fp32_empty)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "0 1 0 0 0 0 0 0 0 4575657222473777152 4575657222473777152 4575657221408423936 4575657222473777152 "
                    "4539628425454813184 ",
                    {0});
}

TEST_F(DecodeBboxV2Tiling, dbv2_reversed_box)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{4, 100}, {4, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 100}, {4, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 100}, {4, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0f, 2.0f, 2.0f, 2.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(5.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "100 1 512 1 2304 1 512 1 100 4611686019501129728 4611686019501129728 4539628425473687552 "
                    "4539628425446424576 4539628425446424576 ",
                    {0});
}

TEST_F(DecodeBboxV2Tiling, dbv2_invalid_rank1)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Tiling, dbv2_invalid_last_dim)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Tiling, dbv2_invalid_shape_mismatch)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 4}, {16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Tiling, dbv2_invalid_dtype_int64)
{
    struct DecodeBboxV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8, 4}, {8, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::TilingContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
