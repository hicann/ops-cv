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

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t index = 0; index < dims.size(); ++index) {
        shape.MutableOriginShape().SetDim(index, dims[index]);
        shape.MutableStorageShape().SetDim(index, dims[index]);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& boxes, const std::vector<int64_t>& queries,
                                        ge::DataType dtype = ge::DT_FLOAT)
{
    const gert::StorageShape boxesShape = MakeStorageShape(boxes);
    const gert::StorageShape queriesShape = MakeStorageShape(queries);
    const gert::StorageShape outputShape = MakeStorageShape({});
    const std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {boxesShape, dtype, ge::FORMAT_ND},
        {queriesShape, dtype, ge::FORMAT_ND},
    };
    const std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {outputShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    return gert::InfershapeContextPara("RotatedOverlaps", inputs, outputs);
}
} // namespace

TEST(RotatedOverlapsInferShape, static_shape)
{
    auto context = MakeContext({2, 5, 7}, {2, 5, 31});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 7, 31}});
}

TEST(RotatedOverlapsInferShape, dynamic_shape)
{
    auto context = MakeContext({-1, 5, -1}, {-1, 5, -1});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, -1, -1}});
}

TEST(RotatedOverlapsInferShape, preserves_old_proto_regression_shape)
{
    auto context = MakeContext({32, 5, 944}, {32, 5, 29});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{32, 944, 29}});
}

TEST(RotatedOverlapsInferShape, rejects_rank_not_three)
{
    auto context = MakeContext({2, 5}, {2, 5, 3});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(RotatedOverlapsInferShape, rejects_channel_not_five)
{
    auto context = MakeContext({2, 4, 7}, {2, 5, 3});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(RotatedOverlapsInferShape, rejects_batch_mismatch)
{
    auto context = MakeContext({2, 5, 7}, {3, 5, 3});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(RotatedOverlapsInferShape, rejects_zero_known_dimension)
{
    auto context = MakeContext({1, 5, 0}, {1, 5, 3});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(RotatedOverlapsInferShape, rejects_query_count_above_first_release_limit)
{
    auto context = MakeContext({1, 5, 1}, {1, 5, 2001});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
