/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grid_sampler2d_grad.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class grid_sampler2d_grad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grid_sampler2d_grad SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grid_sampler2d_grad TearDown" << std::endl;
    }
};

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fp16)
{
    int n = 3;
    int c = 4;
    int h_in = 4;
    int w_in = 5;
    int h_out = 5;
    int w_out = 6;

    gert::InfershapeContextPara infershapeContextPara("GridSampler2DGrad",
                                                      {{{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                       gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                       gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{n, c, h_in, w_in},{n, h_out, w_out, 2},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
