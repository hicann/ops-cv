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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DecodeBboxV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DecodeBboxV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DecodeBboxV2Infershape TearDown" << std::endl; }
};

TEST_F(DecodeBboxV2Infershape, dbv2_fp32_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DecodeBboxV2Infershape, dbv2_fp16_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DecodeBboxV2Infershape, dbv2_fp32_empty)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DecodeBboxV2Infershape, dbv2_reversed_box)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{4, 100}, {4, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 100}, {4, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0f, 2.0f, 2.0f, 2.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(5.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DecodeBboxV2Infershape, dbv2_invalid_rank1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Infershape, dbv2_invalid_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Infershape, dbv2_invalid_shape_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 4}, {16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxV2Infershape, dbv2_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DecodeBboxV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
            gert::InfershapeContextPara::OpAttr("decode_clip", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
