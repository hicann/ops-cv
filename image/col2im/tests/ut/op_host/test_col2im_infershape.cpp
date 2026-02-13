/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_col2im_proto.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class Col2im : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Col2imInfershape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Col2imInfershape TearDown" << std::endl;
    }
};

TEST_F(Col2im, col2im_infershape_fp16)
{
    int n = 8;
    int c = 64;
    int h_col = 22;
    int w_col = 1;
    int w_k = 5;
    int h_k = 1;
    int h = 20;
    int w = 21;

    std::vector<int32_t> inputSizeValues = {h, w};
    gert::InfershapeContextPara infershapeContextPara("Col2im",
                                                      {{{{n, c, w_k*h_k, w_col*h_col}, {n, c, w_k*h_k, w_col*h_col}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},},
                                                      {{{{n, c, h, w}, {n, c, h, w}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("kernel_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 5})),
                                                       gert::InfershapeContextPara::OpAttr("dilation", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 7})),
                                                       gert::InfershapeContextPara::OpAttr("padding", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 5})),
                                                       gert::InfershapeContextPara::OpAttr("stride", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 7}))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{n, c, h, w},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}