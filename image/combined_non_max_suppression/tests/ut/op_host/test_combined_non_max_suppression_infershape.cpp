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
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& boxesShape, const std::vector<int64_t>& scoresShape,
                                        int32_t* maxPerClass, int32_t* maxTotal, bool padPerClass)
{
    const gert::StorageShape boxes = MakeStorageShape(boxesShape);
    const gert::StorageShape scores = MakeStorageShape(scoresShape);
    const gert::StorageShape scalar = MakeStorageShape({});
    const std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {boxes, ge::DT_FLOAT, ge::FORMAT_ND},
        {scores, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_INT32, ge::FORMAT_ND, true, maxPerClass},
        {scalar, ge::DT_INT32, ge::FORMAT_ND, true, maxTotal},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_INT32, ge::FORMAT_ND},
    };
    return gert::InfershapeContextPara(
        "CombinedNonMaxSuppression", inputs, outputs,
        {gert::InfershapeContextPara::OpAttr("pad_per_class", Ops::Cv::AnyValue::CreateFrom<bool>(padPerClass)),
         gert::InfershapeContextPara::OpAttr("clip_boxes", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
}

TEST(CombinedNonMaxSuppressionInfershape, SharedBoxesWithoutPerClassPadding)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, &maxPerClass, &maxTotal, false);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 4, 4}, {2, 4}, {2, 4}, {2}});
}

TEST(CombinedNonMaxSuppressionInfershape, ClassSpecificBoxesWithPerClassPadding)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 10;
    auto context = MakeContext({1, 8, 3, 4}, {1, 8, 3}, &maxPerClass, &maxTotal, true);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 6, 4}, {1, 6}, {1, 6}, {1}});
}

TEST(CombinedNonMaxSuppressionInfershape, RejectsInvalidBoxClassDimension)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    auto context = MakeContext({1, 8, 2, 4}, {1, 8, 3}, &maxPerClass, &maxTotal, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(CombinedNonMaxSuppressionInfershape, RejectsOutputLimitAboveSupportedRange)
{
    int32_t maxPerClass = 1001;
    int32_t maxTotal = 4;
    auto context = MakeContext({1, 8, 1, 4}, {1, 8, 3}, &maxPerClass, &maxTotal, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

} // namespace
