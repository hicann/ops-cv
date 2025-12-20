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
#include "op_proto_test_util.h"
#include "image_ops.h"

class CropAndResize : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CropAndResize SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CropAndResize TearDown" << std::endl;
  }
};

TEST_F(CropAndResize, CropAndResize_infershape_test01){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2,-1, -1, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CropAndResize, CropAndResize_infershape_test02){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test03){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test04){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {}, ge::DT_INT32, ge::FORMAT_ND,
      {}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test05){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {1}, ge::DT_INT32, ge::FORMAT_ND,
      {1}, ge::FORMAT_ND));
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test06){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2, 2}, ge::DT_INT32, ge::FORMAT_ND,
      {2, 2}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test07){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {3, 2, 3, 2}, ge::DT_FLOAT16, ge::FORMAT_HWCN,
      {3, 2, 3, 2}, ge::FORMAT_HWCN));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {3, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {3, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {3}, ge::DT_INT32, ge::FORMAT_ND,
      {3}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_shape_range_0){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_shape_range(
      {2, 3, 4, -1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, -1}, ge::FORMAT_NHWC,{{2, 2}, {3, 3},{4, 4},{1, 9}}));

  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));

  op.UpdateInputDesc("boxes", create_desc_shape_range(
      {-1, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {-1, 4}, ge::FORMAT_ND,{{3, 5},{4, 4}}));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1,-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3, 5}, {0, -1}, {0, -1},{1, 9}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(CropAndResize, CropAndResize_infershape_shape_range_1){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_shape_range(
      {2, -1, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
      {2, -1, 4, 2}, ge::FORMAT_NCHW, {{2, 2}, {9, 15},{4, 4},{2, 2}}));
  op.UpdateInputDesc("boxes", create_desc_shape_range(
      {-1, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {-1, 4}, ge::FORMAT_ND,{{3, 5},{4, 4}}));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1,-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3, 5}, {9, 15}, {0, -1},{0, -1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}