/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {

using OpAttr = gert::InfershapeContextPara::OpAttr;
using TensorDesc = gert::InfershapeContextPara::TensorDescription;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& shape, ge::DataType dtype,
                                        const std::vector<int64_t>& ksize, int64_t borderType)
{
    const auto storageShape = MakeStorageShape(shape);
    const gert::StorageShape outputShape;
    const std::vector<TensorDesc> inputs = {TensorDesc(storageShape, dtype, ge::FORMAT_ND)};
    const std::vector<TensorDesc> outputs = {TensorDesc(outputShape, dtype, ge::FORMAT_ND)};
    const std::vector<OpAttr> attrs = {
        OpAttr("ksize", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)),
        OpAttr("sigma_x", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        OpAttr("sigma_y", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<int64_t>(borderType)),
    };
    return gert::InfershapeContextPara("GaussianBlur", inputs, outputs, attrs);
}

} // namespace

class GaussianBlurInfershapeTest : public testing::Test {};

TEST_F(GaussianBlurInfershapeTest, accepts_gray_and_hwc_images)
{
    auto gray = MakeContext({32, 48}, ge::DT_FLOAT, {3, 3}, 1);
    ExecuteTestCase(gray, ge::GRAPH_SUCCESS, {{32, 48}});

    auto hwc = MakeContext({32, 48, 255}, ge::DT_FLOAT, {31, 31}, 4);
    ExecuteTestCase(hwc, ge::GRAPH_SUCCESS, {{32, 48, 255}});
}

TEST_F(GaussianBlurInfershapeTest, rejects_unsupported_shape_dtype_and_attributes)
{
    auto invalidRank = MakeContext({2, 16, 16, 3}, ge::DT_FLOAT, {3, 3}, 1);
    ExecuteTestCase(invalidRank, ge::GRAPH_FAILED, {{}});

    auto invalidKernel = MakeContext({16, 16, 3}, ge::DT_FLOAT, {4, 3}, 1);
    ExecuteTestCase(invalidKernel, ge::GRAPH_FAILED, {{}});

    auto invalidBorder = MakeContext({16, 16, 3}, ge::DT_FLOAT, {3, 3}, 3);
    ExecuteTestCase(invalidBorder, ge::GRAPH_FAILED, {{}});
}
