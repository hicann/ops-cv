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
 * \file test_roi_pooling_with_arg_max_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RoiPoolingWithArgMax : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "RoiPoolingWithArgMax Infershape Test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "RoiPoolingWithArgMax Infershape Test TearDown" << std::endl;
  }
};

TEST_F(RoiPoolingWithArgMax, RoiPoolingWithArgMax_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara("RoiPoolingWithArgMax",
                                                      {{{{2, 16, 25, 42}, {2, 16, 25, 42}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
                                                       gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
                                                       gert::InfershapeContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(16))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 16, 3, 3}, {2, 16, 3, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RoiPoolingWithArgMax, RoiPoolingWithArgMax_infershape_case_1)
{
    gert::InfershapeContextPara infershapeContextPara("RoiPoolingWithArgMax",
                                                      {{{{1, 8, 10, 10}, {1, 8, 10, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                       gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)),
                                                       gert::InfershapeContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(8))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 8, 2, 2}, {3, 8, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
