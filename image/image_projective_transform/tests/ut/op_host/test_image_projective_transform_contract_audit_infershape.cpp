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
#include <cstdint>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace {

gert::InfershapeContextPara MakeInferShapeCase(int64_t imagesBatch, int64_t transformsBatch)
{
    static std::vector<int32_t> outputShapeData = {2, 3};
    return gert::InfershapeContextPara(
        "ImageProjectiveTransform",
        {{{{imagesBatch, 4, 4, 3}, {imagesBatch, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{transformsBatch, 8}, {transformsBatch, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
}

} // namespace

TEST(ImageProjectiveTransformContractAudit, ValidBatchIsAccepted)
{
    auto context = MakeInferShapeCase(1, 1);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 2, 3, 3}});
}

TEST(ImageProjectiveTransformContractAudit, DynamicBatchIsAcceptedByInferShape)
{
    auto context = MakeInferShapeCase(-1, -1);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 2, 3, 3}});
}

TEST(ImageProjectiveTransformContractAudit, NegativeStaticBatchIsRejectedByInferShape)
{
    for (const auto transformsBatch : {-2, -3, -4}) {
        auto context = MakeInferShapeCase(1, transformsBatch);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(ImageProjectiveTransformContractAudit, NegativeStaticImagesBatchIsRejectedByInferShape)
{
    for (const auto imagesBatch : {-2, -3, -4}) {
        auto context = MakeInferShapeCase(imagesBatch, 1);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(ImageProjectiveTransformContractAudit, UnknownRankInputsRemainAcceptedByInferShape)
{
    static std::vector<int32_t> outputShapeData = {2, 3};
    gert::InfershapeContextPara context("ImageProjectiveTransform",
                                        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}});
}
