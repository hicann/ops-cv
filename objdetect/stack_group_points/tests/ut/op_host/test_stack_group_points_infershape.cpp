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
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_graph/stack_group_points_proto.h"

class StackGroupPoints : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StackGroupPoints Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StackGroupPoints Proto Test TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVectorForStackGroupPoints(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForStackGroupPoints(
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& featuresStorageShape = inputShapes[0];
    const auto& featuresBatchCntStorageShape = inputShapes[1];
    const auto& indicesStorageShape = inputShapes[2];
    const auto& indicesBatchCntStorageShape = inputShapes[3];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType input3Dtype = dtypes[2];
    ge::DataType input4Dtype = dtypes[3];
    ge::DataType outputDtype = dtypes[4];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&featuresStorageShape,
        (gert::Tensor *)&featuresBatchCntStorageShape,
        (gert::Tensor *)&indicesStorageShape,
        (gert::Tensor *)&indicesBatchCntStorageShape,
        (gert::Tensor *)&indicesBatchCntStorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("StackGroupPoints")
        .NodeIoNum(4, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, input3Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, input4Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("StackGroupPoints")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}
// TODO fixme
// TEST_F(StackGroupPoints, StackGroupPoints_infershape_case_0)
// {
//     // 用vector存储同类型参数（顺序与原参数列表一致）
//     std::vector<gert::StorageShape> inputShapes = {
//         {{32, 64}, {32, 64}},    // self_shape
//         {{4,}, {4,}},                  // feeds_shape
//         {{20, 3}, {20, 3}},
//         {{4,}, {4,}},
//         {{4,}, {4,}},
//     };
//     std::vector<ge::DataType> dtypes = {
//         ge::DT_FLOAT,  // input1Dtype
//         ge::DT_INT32,    // input2Dtype
//         ge::DT_INT32,    // input2Dtype
//         ge::DT_INT32,    // input2Dtype
//         ge::DT_FLOAT   // outputDtype
//     };

//     std::vector<int64_t> expectResult = {20, 64, 3};
//     gert::StorageShape outStorageShape = {};
//     // 简化后的函数调用
//     ExeTestCaseForStackGroupPoints(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
//     EXPECT_EQ(ToVectorForStackGroupPoints(outStorageShape.GetOriginShape()), expectResult);

// }