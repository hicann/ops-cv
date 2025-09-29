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
 * \file test_RoiAlignRotated_proto.cpp
 * \brief
 * @version 1.0
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RoiAlignRotated : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "RoiAlignRotated Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "RoiAlignRotated Proto Test TearDown" << std::endl;
  }
};

static std::vector<int64_t> ToVectorForRoiAlignRotated(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForRoiAlignRotated(
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS,
    int64_t pooledH = 2,
    int64_t pooledW = 2,
    float spatialScale = 1.0,
    int64_t samplingRatio = 0,
    bool aligned = true,
    bool clockwise = false)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& xStorageShape = inputShapes[0];
    const auto& roisStorageShape = inputShapes[1];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&xStorageShape,
        (gert::Tensor *)&roisStorageShape,
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("RoiAlignRotated")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("pooled_h", pooledH)
        .Attr("pooled_w", pooledW)
        .Attr("spatial_scale", spatialScale)
        .Attr("sampling_ratio", samplingRatio)
        .Attr("aligned", aligned)
        .Attr("clockwise", clockwise)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("RoiAlignRotated")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{4, 8, 8, 1}, {4, 8, 8, 1}},    // self_shape
        {{6, 8}, {6, 8}},                  // feeds_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT,  // input1Dtype
        ge::DT_FLOAT,    // input2Dtype
    };

    std::vector<int64_t> expectResult = {8, 2, 2, 1};
    gert::StorageShape outStorageShape = {};
    // 简化后的函数调用
    ExeTestCaseForRoiAlignRotated(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS, 2, 2, 1.0, 0, true, false);
    EXPECT_EQ(ToVectorForRoiAlignRotated(outStorageShape.GetOriginShape()), expectResult);
}

// TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_0)
// {
//   ge::op::RoiAlignRotated op;
//   op.UpdateInputDesc("x", create_desc({4, 8, 8, 1}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 8}, ge::DT_FLOAT));
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   std::vector<int64_t> expected_output_shape = {8, 2, 2, 1};
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_1)
// {
//   ge::op::RoiAlignRotated op;
//   op.UpdateInputDesc("x", create_desc({8, 8, 8, 1}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   std::vector<int64_t> expected_output_shape = {3, 2, 2, 1};
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_2)
// {
//   ge::op::RoiAlignRotated op;
//   op.UpdateInputDesc("x", create_desc({8, 8, 8, 8}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   std::vector<int64_t> expected_output_shape = {3, 2, 2, 8};
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotated, RoiAlignRotated_InferDtype_case_0)
// {
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotated"), nullptr);
//   auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotated")->infer_datatype;

//   if (data_type_func != nullptr)
//   {
//     ge::DataType input_ref = ge::DT_FLOAT;
//     ge::DataType rois_ref = ge::DT_FLOAT;
//     ge::DataType output_ref = ge::DT_FLOAT;
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .IrInputNum(2)
//                               .NodeIoNum(2, 1)
//                               .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .InputDataTypes({&input_ref, &rois_ref})
//                               .OutputDataTypes({&output_ref})
//                               .Build();
//     auto context = context_holder.GetContext<gert::InferDataTypeContext>();
//     EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
//     ASSERT_NE(context, nullptr);

//     EXPECT_EQ(context->GetInputDataType(0), input_ref);
//     EXPECT_EQ(context->GetInputDataType(1), rois_ref);
//     EXPECT_EQ(context->GetOutputDataType(0), output_ref);
//   }
// }