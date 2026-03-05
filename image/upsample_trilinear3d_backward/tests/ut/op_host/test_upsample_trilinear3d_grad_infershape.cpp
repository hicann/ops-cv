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
#include <numeric>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class UpsampleTrilinear3dGradTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleTrilinear3dGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleTrilinear3dGradTest TearDown" << std::endl;
    }
};

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test1_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({10, 10, 10})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0, 2.0})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test2_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 10, 10, 10})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test3_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7, 7})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test4_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0, 2.0, 2.0, 2.0})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test5_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5, 1.5, 1.5})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test6_success)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({10, 10, 10})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(UpsampleTrilinear3dGradTest, UpsampleTrilinear3dGrad_infer_test7_success)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleTrilinear3dGrad",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0, 2.0})),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}