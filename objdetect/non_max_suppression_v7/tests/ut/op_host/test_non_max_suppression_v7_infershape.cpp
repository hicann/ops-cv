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
std::vector<gert::InfershapeContextPara::OpAttr> Attrs(int64_t maxBoxes)
{
    return {gert::InfershapeContextPara::OpAttr("center_point_box", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("max_boxes_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxBoxes))};
}
} // namespace

TEST(NonMaxSuppressionV7Infershape, derives_output)
{
    gert::InfershapeContextPara ctx(
        "NonMaxSuppressionV7",
        {{{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 2, 4}, {1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}}, Attrs(3));
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, {{3, 3}});
}

TEST(NonMaxSuppressionV7Infershape, derives_empty_output)
{
    gert::InfershapeContextPara ctx(
        "NonMaxSuppressionV7",
        {{{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 2, 4}, {1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}}, Attrs(0));
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, {{0, 3}});
}

TEST(NonMaxSuppressionV7Infershape, rejects_bad_scores_shape)
{
    gert::InfershapeContextPara ctx(
        "NonMaxSuppressionV7",
        {{{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 2, 5}, {1, 2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}}, Attrs(3));
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}
