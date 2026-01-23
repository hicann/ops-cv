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
 * \file test_resize_bilinear_v2_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ResizeBilinearV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeBilinearV2Infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeBilinearV2Infershape TearDown" << std::endl;
    }
};

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_infer_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-1, -1, -1, 5}, {-1, -1, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_infer_test_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_static_shape_nchw)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_static_shape_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-1, 5, 16, 16}, {-1, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_dy_shape_with_value_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, -1, -1, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_unknownrangk_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_const_size)
{
    std::vector<int32_t> size_data = {2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, size_data.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 2, 2, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_unsupported_input_format)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_fail_case1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ResizeBilinearV2Infershape, resize_bilinear_v2_fail_case2)
{
    std::vector<int32_t> size_data = {2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, size_data.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"dtype", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}