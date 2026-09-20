/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "any_value.h"

class ThreeInterpolateBackwardInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThreeInterpolateBackwardInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ThreeInterpolateBackwardInfershape TearDown" << std::endl; }
};

// Unknown rank: grad_x rank unknown ({-2}), output set to unknown rank
TEST_F(ThreeInterpolateBackwardInfershape, three_interpolate_backward_infershape_unknown_rank_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ThreeInterpolateBackward",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom(int64_t(8))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
