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
 * \file test_grid_sample_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GridSample : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSample SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSample TearDown" << std::endl;
    }
};

TEST_F(GridSample, GridSample_infershape_test_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("GridSample",
                                                      {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TEST_F(GridSample, GridSample_infershape_test_2D_channel_first)
// {
//     ge::op::GridSample op1;
//     ge::DataType xDtype = ge::DT_FLOAT;
//     ge::Format xFormat = ge::FORMAT_NCHW;
//     std::vector<int64_t> xShape = {1, 2, 2, 3};
//     std::vector<std::pair<int64_t, int64_t>> xRange = {{1, 1}, {2, 2}, {2, 2}, {3, 3}};
//     op1.UpdateInputDesc("x", create_desc_shape_range(xShape, xDtype, xFormat, xShape, xFormat, xRange));

//     ge::Format gridFormat = ge::FORMAT_ND;
//     std::vector<int64_t> gridShape = {1, 1, 3, 2};
//     std::vector<std::pair<int64_t, int64_t>> gridRange = {{1, 1}, {2, 2}, {3, 3}, {2, 2}};
//     op1.UpdateInputDesc(
//         "grid", create_desc_shape_range(gridShape, xDtype, gridFormat, gridShape, gridFormat, gridRange));
//     op1.SetAttr("channel_last", false);
//     std::vector<int64_t> expectShape = {1, 2, 1, 3};

//     Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners", "channel_last"}};
//     EXPECT_EQ(InferShapeTest(op1, param1), ge::GRAPH_SUCCESS);

//     auto yDesc = op1.GetOutputDescByName("y");
//     EXPECT_EQ(yDesc.GetDataType(), xDtype);
//     EXPECT_EQ(yDesc.GetShape().GetDims(), expectShape);
// }

// TEST_F(GridSample, GridSample_infershape_test_2D_channel_last_unknown_dim)
// {
//     ge::op::GridSample op1;
//     ge::DataType xDtype = ge::DT_FLOAT;
//     ge::Format xFormat = ge::FORMAT_NHWC;
//     std::vector<int64_t> xShape = {1, -1, 2, 3};
//     std::vector<std::pair<int64_t, int64_t>> xRange = {{1, 1}, {2, 2}, {2, 2}, {3, 3}};
//     op1.UpdateInputDesc("x", create_desc_shape_range(xShape, xDtype, xFormat, xShape, xFormat, xRange));

//     ge::Format gridFormat = ge::FORMAT_ND;
//     std::vector<int64_t> gridShape = {1, 1, -1, 2};
//     std::vector<std::pair<int64_t, int64_t>> gridRange = {{1, 1}, {2, 2}, {3, 3}, {2, 2}};
//     op1.UpdateInputDesc(
//         "grid", create_desc_shape_range(gridShape, xDtype, gridFormat, gridShape, gridFormat, gridRange));
//     op1.SetAttr("channel_last", true);
//     std::vector<int64_t> expectShape = {1, 3, 1, -1};

//     Runtime2TestParam param1{{"interpolation_mode", "padding_mode", "align_corners", "channel_last"}};
//     EXPECT_EQ(InferShapeTest(op1, param1), ge::GRAPH_SUCCESS);

//     auto yDesc = op1.GetOutputDescByName("y");
//     EXPECT_EQ(yDesc.GetDataType(), xDtype);
//     EXPECT_EQ(yDesc.GetShape().GetDims(), expectShape);
// }

// TEST_F(GridSample, GridSample_infershaperange_test_2D_channel_first)
// {
//     gert::Shape x_range_max{4, 6, 32, 32};
//     gert::Shape x_range_min{2, 3, 16, 16};
//     gert::Shape grid_range_max{4, 44, 55, 2};
//     gert::Shape grid_range_min{2, 22, 33, 2};
//     gert::Shape y_range_max{4, 6, 44, 55};
//     gert::Shape y_range_min{2, 3, 22, 33};

//     gert::Range<gert::Shape> x_range(&x_range_min, &x_range_max);
//     gert::Range<gert::Shape> grid_range(&grid_range_min, &grid_range_max);
//     gert::Range<gert::Shape> expected_output_range(&y_range_min, &y_range_max);

//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample"), nullptr);
//     auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample")->infer_shape_range;
//     ASSERT_NE(infer_func, nullptr);

//     auto context_holder = gert::InferShapeRangeContextFaker()
//                               .IrInputNum(2)
//                               .NodeIoNum(2, 1)
//                               .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeAttrs({
//                                   {"interpolation_mode", ge::AnyValue::CreateFrom<string>("bilinear")},
//                                   {"padding_mode", ge::AnyValue::CreateFrom<string>("zeros")},
//                                   {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                                   {"channel_last", ge::AnyValue::CreateFrom<bool>(false)},
//                                   {"scheduler_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
//                               })
//                               .InputShapeRanges({&x_range, &grid_range})
//                               .OutputShapeRanges({&expected_output_range})
//                               .Build();

//     auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
//     EXPECT_EQ(infer_func(context), ge::GRAPH_SUCCESS);

//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 2);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 3);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(2), 22);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(3), 33);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 4);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 6);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(2), 44);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(3), 55);
// }

// TEST_F(GridSample, GridSample_infershaperange_test_2D_channel_last)
// {
//     gert::Shape x_range_max{4, 32, 32, 6};
//     gert::Shape x_range_min{2, 16, 16, 3};
//     gert::Shape grid_range_max{4, 44, 55, 2};
//     gert::Shape grid_range_min{2, 22, 33, 2};
//     gert::Shape y_range_max{4, 6, 44, 55};
//     gert::Shape y_range_min{2, 3, 22, 33};

//     gert::Range<gert::Shape> x_range(&x_range_min, &x_range_max);
//     gert::Range<gert::Shape> grid_range(&grid_range_min, &grid_range_max);
//     gert::Range<gert::Shape> expected_output_range(&y_range_min, &y_range_max);

//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample"), nullptr);
//     auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample")->infer_shape_range;
//     ASSERT_NE(infer_func, nullptr);

//     auto context_holder = gert::InferShapeRangeContextFaker()
//                               .IrInputNum(2)
//                               .NodeIoNum(2, 1)
//                               .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeAttrs({
//                                   {"interpolation_mode", ge::AnyValue::CreateFrom<string>("bilinear")},
//                                   {"padding_mode", ge::AnyValue::CreateFrom<string>("zeros")},
//                                   {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                                   {"channel_last", ge::AnyValue::CreateFrom<bool>(true)},
//                                   {"scheduler_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
//                               })
//                               .InputShapeRanges({&x_range, &grid_range})
//                               .OutputShapeRanges({&expected_output_range})
//                               .Build();

//     auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
//     EXPECT_EQ(infer_func(context), ge::GRAPH_SUCCESS);

//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 2);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 3);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(2), 22);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(3), 33);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 4);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 6);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(2), 44);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(3), 55);
// }

// TEST_F(GridSample, GridSample_infershaperange_test_unkown_rank)
// {
//     gert::Shape x_range_max{7};
//     gert::Shape x_range_min{3};
//     gert::Shape grid_range_max{4, 44, 55, 2};
//     gert::Shape grid_range_min{2, 22, 33, 2};
//     gert::Shape y_range_max{7};
//     gert::Shape y_range_min{3};

//     gert::Range<gert::Shape> x_range(&x_range_min, &x_range_max);
//     gert::Range<gert::Shape> grid_range(&grid_range_min, &grid_range_max);
//     gert::Range<gert::Shape> expected_output_range(&y_range_min, &y_range_max);

//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample"), nullptr);
//     auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GridSample")->infer_shape_range;
//     ASSERT_NE(infer_func, nullptr);

//     auto context_holder = gert::InferShapeRangeContextFaker()
//                               .IrInputNum(2)
//                               .NodeIoNum(2, 1)
//                               .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                               .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
//                               .NodeAttrs({
//                                   {"interpolation_mode", ge::AnyValue::CreateFrom<string>("bilinear")},
//                                   {"padding_mode", ge::AnyValue::CreateFrom<string>("zeros")},
//                                   {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                                   {"channel_last", ge::AnyValue::CreateFrom<bool>(true)},
//                                   {"scheduler_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
//                               })
//                               .InputShapeRanges({&x_range, &grid_range})
//                               .OutputShapeRanges({&expected_output_range})
//                               .Build();

//     auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
//     EXPECT_EQ(infer_func(context), ge::GRAPH_SUCCESS);

//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 3);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 7);
// }
