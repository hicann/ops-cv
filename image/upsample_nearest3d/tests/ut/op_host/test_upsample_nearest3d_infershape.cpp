/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class UpsampleNearest3dTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleNearest3dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleNearest3dTest TearDown" << std::endl;
    }
};

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test1_failed)
{
    gert::InfershapeContextPara infershapeContextPara("UpsampleNearest3d",
                                                      {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {gert::InfershapeContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
                                                       gert::InfershapeContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0, 2.0}))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
