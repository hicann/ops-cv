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
 * \file test_roi_pooling_grad_with_arg_max_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RoiPoolingGradWithArgMax : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RoiPoolingGradWithArgMaxInfershape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RoiPoolingGradWithArgMaxInfershape TearDown" << std::endl;
    }
};

TEST_F(RoiPoolingGradWithArgMax, roi_pooling_grad_with_arg_max_infershape_fp16)
{
    int n = 1;
    int c = 32;
    int poolh = 2;
    int poolw = 2;
    int height = 3;
    int width = 3;
    int rois_n = 4;

    gert::InfershapeContextPara infershapeContextPara("RoiPoolingGradWithArgMax",
                                                      {{{{rois_n, c, poolh, poolw}, {n, c, poolh, poolw}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{n, c, height, width}, {n, c, height, width}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{rois_n, 5}, {rois_n, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{rois_n, 5}, {rois_n, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      {{{rois_n, c, poolh, poolw}, {n, c, poolh, poolw}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{n, c, height, width}, {n, c, height, width}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolh)),
                                                       gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolw)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
                                                       gert::InfershapeContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(c))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{n, c, height, width},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}