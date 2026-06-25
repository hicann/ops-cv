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
#include <stdint.h>
#include <iostream>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_kernel/arch35/image_projective_transform_tiling_data.h"
#include "../../../op_kernel/arch35/image_projective_transform_tiling_key.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ImageProjectiveTransformTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ImageProjectiveTransformTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ImageProjectiveTransformTiling TearDown" << std::endl; }
};

struct ImageProjectiveTransformCompileInfo {};

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float32_bilinear)
{
    int64_t N = 1;
    int64_t HIn = 4;
    int64_t WIn = 4;
    int64_t C = 3;
    int64_t HOut = 4;
    int64_t WOut = 4;
    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
    std::initializer_list<int64_t> transformsShape = {N, 8};
    std::initializer_list<int64_t> outputShapeShape = {2};
    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ImageProjectiveTransform",
                                                {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                {{{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
                                                {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
                                                gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
                                                &compileInfo);
        uint64_t expectTilingKey = 0;
        std::vector<size_t> expectWorkspaces = {16777216};
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float32_nearest)
{
        int64_t N = 1;
        int64_t HIn = 4;
        int64_t WIn = 4;
        int64_t C = 3;
        int64_t HOut = 4;
        int64_t WOut = 4;
        std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
        std::initializer_list<int64_t> transformsShape = {N, 8};
        std::initializer_list<int64_t> outputShapeShape = {2};
        std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
        ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ImageProjectiveTransform",
                                                {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                {{{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
                                                {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("NEAREST")),
                                                gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
                                                &compileInfo);
            uint64_t expectTilingKey = 1;
            std::vector<size_t> expectWorkspaces = {16777216};
            ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_float16_bilinear)
{
            int64_t N = 1;
            int64_t HIn = 4;
            int64_t WIn = 4;
            int64_t C = 3;
            int64_t HOut = 4;
            int64_t WOut = 4;
            std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
            std::initializer_list<int64_t> transformsShape = {N, 8};
            std::initializer_list<int64_t> outputShapeShape = {2};
            std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
            ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ImageProjectiveTransform",
                                                {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
                                                {{{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
                                                {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
                                                gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
                                                &compileInfo);
                uint64_t expectTilingKey = 0;
                std::vector<size_t> expectWorkspaces = {16777216};
                ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_uint8_bilinear)
{
                int64_t N = 1;
                int64_t HIn = 4;
                int64_t WIn = 4;
                int64_t C = 3;
                int64_t HOut = 4;
                int64_t WOut = 4;
                std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
                std::initializer_list<int64_t> transformsShape = {N, 8};
                std::initializer_list<int64_t> outputShapeShape = {2};
                std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
                ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ImageProjectiveTransform",
                                                {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_UINT8, ge::FORMAT_NHWC},
                                                {{{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
                                                {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("BILINEAR")),
                                                gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
                                                &compileInfo);
                    uint64_t expectTilingKey = 0;
                    std::vector<size_t> expectWorkspaces = {16777216};
                    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_test_int32_nearest)
{
                    int64_t N = 1;
                    int64_t HIn = 4;
                    int64_t WIn = 4;
                    int64_t C = 3;
                    int64_t HOut = 4;
                    int64_t WOut = 4;
                    std::initializer_list<int64_t> imagesShape = {N, HIn, WIn, C};
                    std::initializer_list<int64_t> transformsShape = {N, 8};
                    std::initializer_list<int64_t> outputShapeShape = {2};
                    std::initializer_list<int64_t> outShape = {N, HOut, WOut, C};
                    ImageProjectiveTransformCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ImageProjectiveTransform",
                                                {{{{N, HIn, WIn, C}, {N, HIn, WIn, C}}, ge::DT_INT32, ge::FORMAT_NHWC},
                                                {{{{N, 8}, {N, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                {{{{N, HOut, WOut, C}, {N, HOut, WOut, C}}, ge::DT_INT32, ge::FORMAT_NHWC}},
                                                {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<string>("NEAREST")),
                                                gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<string>("CONSTANT"))},
                                                &compileInfo);
                        uint64_t expectTilingKey = 1;
                        std::vector<size_t> expectWorkspaces = {16777216};
                        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}
