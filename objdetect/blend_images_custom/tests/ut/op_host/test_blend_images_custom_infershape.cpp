/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class BlendImagesCustom : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BlendImagesCustom Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BlendImagesCustom Proto Test TearDown" << std::endl; }
};

TEST_F(BlendImagesCustom, BlendImagesCustom_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara("BlendImagesCustom",
                                                      {{{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                       {{{480, 640, 1}, {480, 640, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                       {{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {480, 640, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BlendImagesCustom, BlendImagesCustom_infershape_case_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BlendImagesCustom",
        {{{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{1080, 1920, 1}, {1080, 1920, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1080, 1920, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}