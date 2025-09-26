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
#include "op_proto_test_util.h"
#include "common/utils/ut_op_common.h"
#include "nn_detect.h"

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

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_0)
{
  ge::op::RoiAlignRotated op;
  op.UpdateInputDesc("x", create_desc({4, 8, 8, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("rois", create_desc({6, 8}, ge::DT_FLOAT));
  int32_t pooled_h = 2;
  int32_t pooled_w = 2;
  float32_t spatial_scale = 1;
  int32_t sampling_ratio = 0;
  bool aligned = true;
  bool clockwise = false;

  op.SetAttr("pooled_h", pooled_h);
  op.SetAttr("pooled_w", pooled_w);
  op.SetAttr("spatial_scale", spatial_scale);
  op.SetAttr("sampling_ratio", sampling_ratio);
  op.SetAttr("aligned", aligned);
  op.SetAttr("clockwise", clockwise);
  Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
  EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {8, 2, 2, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_1)
{
  ge::op::RoiAlignRotated op;
  op.UpdateInputDesc("x", create_desc({8, 8, 8, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
  int32_t pooled_h = 2;
  int32_t pooled_w = 2;
  float32_t spatial_scale = 1;
  int32_t sampling_ratio = 0;
  bool aligned = true;
  bool clockwise = false;

  op.SetAttr("pooled_h", pooled_h);
  op.SetAttr("pooled_w", pooled_w);
  op.SetAttr("spatial_scale", spatial_scale);
  op.SetAttr("sampling_ratio", sampling_ratio);
  op.SetAttr("aligned", aligned);
  op.SetAttr("clockwise", clockwise);
  Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
  EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {3, 2, 2, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_2)
{
  ge::op::RoiAlignRotated op;
  op.UpdateInputDesc("x", create_desc({8, 8, 8, 8}, ge::DT_FLOAT));
  op.UpdateInputDesc("rois", create_desc({6, 3}, ge::DT_FLOAT));
  int32_t pooled_h = 2;
  int32_t pooled_w = 2;
  float32_t spatial_scale = 1;
  int32_t sampling_ratio = 0;
  bool aligned = true;
  bool clockwise = false;

  op.SetAttr("pooled_h", pooled_h);
  op.SetAttr("pooled_w", pooled_w);
  op.SetAttr("spatial_scale", spatial_scale);
  op.SetAttr("sampling_ratio", sampling_ratio);
  op.SetAttr("aligned", aligned);
  op.SetAttr("clockwise", clockwise);
  Runtime2TestParam roi_align_rotated_param{{"pooled_h", "pooled_w", "spatial_scale", "sampling_ratio", "aligned", "clockwise"}};
  EXPECT_EQ(InferShapeTest(op, roi_align_rotated_param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {3, 2, 2, 8};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_InferDtype_case_0)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotated"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RoiAlignRotated")->infer_datatype;

  if (data_type_func != nullptr)
  {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType rois_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_ref, &rois_ref})
                              .OutputDataTypes({&output_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetInputDataType(1), rois_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}