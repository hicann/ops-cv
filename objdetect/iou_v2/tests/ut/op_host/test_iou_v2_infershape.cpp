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

class IouV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "IouV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IouV2 Proto Test TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVectorForIouV2(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForIouV2(
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS,
    const ge::AscendString& modeAttr = "iou",
    float epsAttr = 1.0,
    bool alignedAttr = false)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& bboxesStorageShape = inputShapes[0];
    const auto& gtboxesStorageShape = inputShapes[1];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&bboxesStorageShape,
        (gert::Tensor *)&gtboxesStorageShape,
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("IouV2")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("mode", modeAttr)
        .Attr("eps", epsAttr)
        .Attr("aligned", alignedAttr)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("IouV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(IouV2, IouV2_infershape_iou_false_case_0)
{
    // size_t size1 = 4;
    // size_t size2 = 5;
    // size_t size3 = 6;
    // size_t size4 = 7;
    // size_t feeds_size = 4;
    // size_t out_size = 15;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{1024, 4}, {1024, 4}},    // self_shape
        {{1024, 4}, {1024, 4}},                  // feeds_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT,  // input1Dtype
        ge::DT_FLOAT,    // input2Dtype
    };

    std::vector<int64_t> expectResult = {1024, 1024};
    gert::StorageShape outStorageShape = {};
    // 简化后的函数调用
    ExeTestCaseForIouV2(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS, "iou", 1.0, false);
    EXPECT_EQ(ToVectorForIouV2(outStorageShape.GetOriginShape()), expectResult);
}

// TEST_F(IouV2, IouV2_infershape_iof_false_case_2)
// {
//     ge::op::IouV2 op;
//     float32_t eps_value = 1;
//     op.UpdateInputDesc("bboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.UpdateInputDesc("gtboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.SetAttr("mode", "iof");
//     op.SetAttr("eps", eps_value);
//     op.SetAttr("aligned", false);
//     Runtime2TestParam param{{"mode", "eps", "aligned"}, {}, {}};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

//     auto output_desc = op.GetOutputDesc("overlap");
//     std::vector<int64_t> expected_output_shape = {1020, 1020};
//     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(IouV2, IouV2_infershape_iou_true_case_3)
// {
//     ge::op::IouV2 op;
//     float32_t eps_value = 1;
//     op.UpdateInputDesc("bboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.UpdateInputDesc("gtboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.SetAttr("mode", "iou");
//     op.SetAttr("eps", eps_value);
//     op.SetAttr("aligned", true);
//     Runtime2TestParam param{{"mode", "eps", "aligned"}, {}, {}};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

//     auto output_desc = op.GetOutputDesc("overlap");
//     std::vector<int64_t> expected_output_shape = {4, 1};
//     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(IouV2, IouV2_InferDtype_case_0)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("IouV2"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("IouV2")->infer_datatype;
//     if (data_type_func != nullptr)
//     {
//         ge::DataType bboxes = ge::DT_FLOAT;
//         ge::DataType gtboxes = ge::DT_FLOAT;
//         ge::DataType output_ref = ge::DT_FLOAT;
//         auto context_holder = gert::InferDataTypeContextFaker()
//                                   .IrInputNum(2)
//                                   .NodeIoNum(2, 1)
//                                   .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .InputDataTypes({&bboxes, &gtboxes})
//                                   .OutputDataTypes({&output_ref})
//                                   .Build();
//         auto context = context_holder.GetContext<gert::InferDataTypeContext>();
//         EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
//         ASSERT_NE(context, nullptr);

//         EXPECT_EQ(context->GetInputDataType(0), bboxes);
//         EXPECT_EQ(context->GetInputDataType(1), gtboxes);
//         EXPECT_EQ(context->GetOutputDataType(0), output_ref);
//     }
// }