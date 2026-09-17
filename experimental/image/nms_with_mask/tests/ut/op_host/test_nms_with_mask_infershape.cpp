/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
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

using TensorDesc = gert::InfershapeContextPara::TensorDescription;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& xDims)
{
    const auto xShape = MakeStorageShape(xDims);
    const auto scoreShape = MakeStorageShape({2});
    const gert::StorageShape outputShape;
    return gert::InfershapeContextPara(
        "NMSWithMask",
        {TensorDesc(xShape, ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(scoreShape, ge::DT_FLOAT, ge::FORMAT_ND)},
        {TensorDesc(outputShape, ge::DT_UINT8, ge::FORMAT_ND)});
}

} // namespace

TEST(NMSWithMaskInfershapeValidation, acceptsValidBoxShape)
{
    auto context = MakeContext({2, 4});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2}});
}

TEST(NMSWithMaskInfershapeValidation, acceptsUnknownRank)
{
    auto context = MakeContext({ge::UNKNOWN_RANK[0]});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{ge::UNKNOWN_RANK[0]}});
}

TEST(NMSWithMaskInfershapeValidation, acceptsUnknownCoordinateDimension)
{
    auto context = MakeContext({2, ge::UNKNOWN_DIM});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2}});
}

TEST(NMSWithMaskInfershapeValidation, rejectsRankZero)
{
    auto context = MakeContext({});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(NMSWithMaskInfershapeValidation, rejectsWrongRank)
{
    auto rankOneContext = MakeContext({2});
    ExecuteTestCase(rankOneContext, ge::GRAPH_FAILED);

    auto rankThreeContext = MakeContext({2, 4, 1});
    ExecuteTestCase(rankThreeContext, ge::GRAPH_FAILED);
}

TEST(NMSWithMaskInfershapeValidation, rejectsWrongCoordinateDimension)
{
    auto context = MakeContext({2, 5});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
