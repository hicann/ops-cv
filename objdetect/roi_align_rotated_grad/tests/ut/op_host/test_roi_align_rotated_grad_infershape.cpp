/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_RoiAlignRotatedGrad_proto.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RoiAlignRotatedGrad : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "RoiAlignRotatedGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "RoiAlignRotatedGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(RoiAlignRotatedGrad, RoiAlignRotatedGrad_infershape_iou_false_case_0)
{
    gert::InfershapeContextPara infershapeContextPara("RoiAlignRotatedGrad",
                                                      {{{{8, 2, 2, 1}, {8, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("y_grad_shape", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({4, 8, 8, 1})),
                                                       gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                       gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                       gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
                                                       gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
                                                       gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                       gert::InfershapeContextPara::OpAttr("clockwise", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 8, 1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TEST_F(RoiAlignRotatedGrad, RoiAlignRotatedGrad_infershape_case_0)
// {
//   ge::op::RoiAlignRotatedGrad op;
//   op.UpdateInputDesc("x_grad", create_desc({8, 2, 2, 1}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 8}, ge::DT_FLOAT));
//   std::vector<int64_t> expected_output_shape = {4, 8, 8, 1};
//   std::vector<int64_t> y_grad_shape = {4, 8, 8, 1};
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("y_grad_shape", y_grad_shape);
//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_grad_param{{"y_grad_shape", "pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_grad_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotatedGrad, RoiAlignRotatedGrad_infershape_case_1)
// {
//   ge::op::RoiAlignRotatedGrad op;
//   op.UpdateInputDesc("x_grad", create_desc({3, 2, 2, 1}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
//   std::vector<int64_t> expected_output_shape = {8, 8, 8, 1};
//   std::vector<int64_t> y_grad_shape = {8, 8, 8, 1};
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("y_grad_shape", y_grad_shape);
//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_grad_param{{"y_grad_shape", "pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_grad_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotatedGrad, RoiAlignRotatedGrad_infershape_case_2)
// {
//   ge::op::RoiAlignRotatedGrad op;
//   op.UpdateInputDesc("x_grad", create_desc({3, 2, 2, 8}, ge::DT_FLOAT));
//   op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
//   std::vector<int64_t> expected_output_shape = {8, 8, 8, 8};
//   std::vector<int64_t> y_grad_shape = {8, 8, 8, 8};
//   int32_t pooled_h = 2;
//   int32_t pooled_w = 2;
//   float32_t spatial_scale = 1;
//   int32_t sampling_ratio = 0;
//   bool aligned = true;
//   bool clockwise = false;

//   op.SetAttr("y_grad_shape", y_grad_shape);
//   op.SetAttr("pooled_h", pooled_h);
//   op.SetAttr("pooled_w", pooled_w);
//   op.SetAttr("spatial_scale", spatial_scale);
//   op.SetAttr("sampling_ratio", sampling_ratio);
//   op.SetAttr("aligned", aligned);
//   op.SetAttr("clockwise", clockwise);
//   Runtime2TestParam roi_align_rotated_grad_param{{"y_grad_shape", "pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
//   EXPECT_EQ(InferShapeTest(op, roi_align_rotated_grad_param), ge::GRAPH_SUCCESS);

//   auto output_desc = op.GetOutputDesc(0);
//   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(RoiAlignRotatedGrad, RoiAlignRotatedGrad_InferDtype_case_0)
// {
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotatedGrad"), nullptr);
//   auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotatedGrad")->infer_datatype;

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
