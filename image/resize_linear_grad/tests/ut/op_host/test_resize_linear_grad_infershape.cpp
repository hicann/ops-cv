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
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ResizeLinearGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeLinearGradInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeLinearGradInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(ResizeLinearGradInfershapeTest, resize_linear_grad_infershape_test_01)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape outputShape = {{-2,}, {-2,}};

    gert::InfershapeContextPara infershapeContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, 32},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}