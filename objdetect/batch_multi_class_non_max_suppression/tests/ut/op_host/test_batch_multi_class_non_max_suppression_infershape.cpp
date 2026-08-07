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
#include <vector>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
std::vector<gert::InfershapeContextPara::OpAttr> GetAttrs(int64_t maxTotalSize)
{
    return {
        gert::InfershapeContextPara::OpAttr("score_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.3F)),
        gert::InfershapeContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5F)),
        gert::InfershapeContextPara::OpAttr("max_size_per_class", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
        gert::InfershapeContextPara::OpAttr("max_total_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxTotalSize)),
        gert::InfershapeContextPara::OpAttr("change_coordinate_frame", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        gert::InfershapeContextPara::OpAttr("transpose_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
    };
}
} // namespace

TEST(BatchMultiClassNonMaxSuppressionInfershape, derives_public_output_layout)
{
    gert::InfershapeContextPara context("BatchMultiClassNonMaxSuppression",
                                        {{{{2, 8, 1, 4}, {2, 8, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 8, 3}, {2, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                        GetAttrs(5));
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 5, 4}, {2, 5}, {2, 5}, {2}});
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, rejects_non_rank_four_boxes)
{
    gert::InfershapeContextPara context(
        "BatchMultiClassNonMaxSuppression",
        {{{{2, 8, 4}, {2, 8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 8, 3}, {2, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        GetAttrs(5));
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
