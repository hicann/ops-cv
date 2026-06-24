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

class ResizeNearestNeighborV2GradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeNearestNeighborV2GradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeNearestNeighborV2GradInfershapeTest TearDown" << std::endl; }
};

TEST_F(ResizeNearestNeighborV2GradInfershapeTest, resize_nearest_neighbor_v2_grad_infershape_test_01)
{
    gert::StorageShape inputXShape = {{1, 2, 3, 32}, {1, 2, 3, 32}};
    gert::StorageShape inputSizeShape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape outputShape = {{1, 2, 6, 64}, {1, 2, 6, 64}};
    int size_value[2] = {6, 64};

    gert::InfershapeContextPara infershapeContextPara(
        "ResizeNearestNeighborV2Grad",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 6, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeNearestNeighborV2GradInfershapeTest, resize_nearest_neighbor_v2_grad_infershape_test_02)
{
    gert::StorageShape inputXShape = {{1, 8, 10, 16}, {1, 8, 10, 16}};
    gert::StorageShape inputSizeShape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape outputShape = {{1, 8, 5, 8}, {1, 8, 5, 8}};
    int size_value[2] = {5, 8};

    gert::InfershapeContextPara infershapeContextPara(
        "ResizeNearestNeighborV2Grad",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 8, 5, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeNearestNeighborV2GradInfershapeTest, resize_nearest_neighbor_v2_grad_infershape_test_03_nhwc)
{
    gert::StorageShape inputXShape = {{1, 10, 16, 8}, {1, 10, 16, 8}};
    gert::StorageShape inputSizeShape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape outputShape = {{1, 5, 8, 8}, {1, 5, 8, 8}};
    int size_value[2] = {5, 8};

    gert::InfershapeContextPara infershapeContextPara("ResizeNearestNeighborV2Grad",
                                                      {{inputXShape, ge::DT_FLOAT16, ge::FORMAT_NHWC},
                                                       {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
                                                      {{outputShape, ge::DT_FLOAT16, ge::FORMAT_NHWC}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 5, 8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeNearestNeighborV2GradInfershapeTest, resize_nearest_neighbor_v2_grad_infershape_test_04_same_size)
{
    gert::StorageShape inputXShape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape inputSizeShape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape outputShape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    int size_value[2] = {4, 4};

    gert::InfershapeContextPara infershapeContextPara(
        "ResizeNearestNeighborV2Grad",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeNearestNeighborV2GradInfershapeTest, resize_nearest_neighbor_v2_grad_infershape_test_05_bf16)
{
    gert::StorageShape inputXShape = {{1, 4, 64, 64}, {1, 4, 64, 64}};
    gert::StorageShape inputSizeShape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape outputShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    int size_value[2] = {32, 32};

    gert::InfershapeContextPara infershapeContextPara(
        "ResizeNearestNeighborV2Grad",
        {{inputXShape, ge::DT_BF16, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_BF16, ge::FORMAT_NCHW}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 4, 32, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
