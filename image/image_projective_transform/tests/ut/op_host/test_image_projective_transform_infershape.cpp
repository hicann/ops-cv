/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ImageProjectiveTransformInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ImageProjectiveTransformInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ImageProjectiveTransformInfershape TearDown" << std::endl; }
};

TEST_F(ImageProjectiveTransformInfershape, output_shape_const_override_h_w)
{
    int64_t N = 2;
    int64_t HIn = 4;
    int64_t WIn = 5;
    int64_t C = 3;
    int64_t HOut = 8;
    int64_t WOut = 9;
    std::vector<int32_t> outputShapeData = {static_cast<int32_t>(HOut), static_cast<int32_t>(WOut)};
    gert::InfershapeContextPara infershapeContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{N, HOut, WOut, C}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, output_shape_nonconst_produces_unknown_hw)
{
    int64_t N = 2;
    int64_t HIn = 4;
    int64_t WIn = 5;
    int64_t C = 3;
    gert::InfershapeContextPara infershapeContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{N, -1, -1, C}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, output_shape_const_zero_preserves_empty_dimension)
{
    int64_t N = 2;
    int64_t HIn = 4;
    int64_t WIn = 5;
    int64_t C = 3;
    std::vector<int32_t> outputShapeData = {0, static_cast<int32_t>(WIn)};
    gert::InfershapeContextPara infershapeContextPara(
        "ImageProjectiveTransform",
        {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{N, 0, WIn, C}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, dynamic_batch_dim)
{
    int64_t HIn = 4;
    int64_t WIn = 5;
    int64_t C = 3;
    int64_t HOut = 8;
    int64_t WOut = 9;
    std::vector<int32_t> outputShapeData = {static_cast<int32_t>(HOut), static_cast<int32_t>(WOut)};
    gert::InfershapeContextPara infershapeContextPara(
        "ImageProjectiveTransform",
        {{{{-1, HIn, WIn, C}, {-1, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{-1, 8}, {-1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, HOut, WOut, C}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, images_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, transforms_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{2, 4, 5, 3}, {2, 4, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, output_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{2, 4, 5, 3}, {2, 4, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, images_unknown_rank_does_not_hide_invalid_transforms_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{2, 8, 1}, {2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ImageProjectiveTransformInfershape, transforms_unknown_rank_does_not_hide_invalid_images_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ImageProjectiveTransformInfershape, output_shape_size_3_is_invalid)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{2, 4, 5, 3}, {2, 4, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ImageProjectiveTransformInfershape, unknown_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ImageProjectiveTransform",
        {{{{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ImageProjectiveTransformInfershape, invalid_rank_3d)
{
    gert::InfershapeContextPara infershapeContextPara("ImageProjectiveTransform",
                                                      {{{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                       {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
