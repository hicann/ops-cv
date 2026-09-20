/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rotated_feature_align_grad_infershape.cpp
 * \brief UT for rotated_feature_align_grad infershape
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RotatedFeatureAlignGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RotatedFeatureAlignGrad InferShape Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RotatedFeatureAlignGrad InferShape Test TearDown" << std::endl; }
};

TEST_F(RotatedFeatureAlignGradInferShape, infershape_pts1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotatedFeatureAlignGradInferShape, infershape_pts5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{1, 8, 8, 16}, {1, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 8, 5}, {1, 8, 8, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 8, 8, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Unknown rank(-2)：dy/bboxes 均为 {-2}，输出置 unknown rank {-2}
TEST_F(RotatedFeatureAlignGradInferShape, infershape_unknown_rank_both)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// bboxes 为 unknown rank(-2)，dy 已知：跳过 bboxes 校验，输出取 dy shape
TEST_F(RotatedFeatureAlignGradInferShape, infershape_unknown_rank_bboxes_only)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Unknown dim(-1)：dy/bboxes 的 N/H 维为 -1，一致性校验跳过，输出 -1 透传
TEST_F(RotatedFeatureAlignGradInferShape, infershape_unknown_dim_nhw)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{-1, 4, -1, 3}, {-1, 4, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{-1, 4, -1, 5}, {-1, 4, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 4, -1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// bboxes 末维为 unknown dim(-1)：跳过末维=5 校验，输出取 dy shape
TEST_F(RotatedFeatureAlignGradInferShape, infershape_unknown_dim_bboxes_last)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, -1}, {2, 4, 4, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dy 与 bboxes 的已知维冲突（H: 4 vs 5）：仍需报错
TEST_F(RotatedFeatureAlignGradInferShape, infershape_known_dim_mismatch_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 5, 4, 5}, {2, 5, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// dy 非 4D：触发 OP_LOGE_FOR_INVALID_SHAPEDIM，返回失败
TEST_F(RotatedFeatureAlignGradInferShape, infershape_fail_dy_shapedim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// bboxes 非 4D：触发 OP_LOGE_FOR_INVALID_SHAPEDIM，返回失败
TEST_F(RotatedFeatureAlignGradInferShape, infershape_fail_bboxes_shapedim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// bboxes 末维非 5：触发 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON，返回失败
TEST_F(RotatedFeatureAlignGradInferShape, infershape_fail_bboxes_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 6}, {2, 4, 4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// points 非 1/5：触发 OP_LOGE_FOR_INVALID_VALUE，返回失败
TEST_F(RotatedFeatureAlignGradInferShape, infershape_fail_points)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::InfershapeContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(3))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
