/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/grid_sample_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

struct GridSampleDtypeMismatchCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool regBase = false;
};

TEST(GridSampleDtypeMismatch, MatchingDtype2DIsAccepted)
{
    GridSampleDtypeMismatchCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSample",
        {{{{2, 200, 200, 1}, {2, 200, 200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 2, 1}, {2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 1000220;
    string expectTilingData = "48 2 1 0 200 200 0 2 2 0 0 1 1 2 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST(GridSampleDtypeMismatch, MismatchedDtype2DIsRejected)
{
    GridSampleDtypeMismatchCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSample",
        {{{{2, 200, 200, 1}, {2, 200, 200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 2, 2, 1}, {2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST(GridSampleDtypeMismatch, MismatchedDtype3DIsRejected)
{
    GridSampleDtypeMismatchCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSample",
        {{{{2, 200, 200, 200, 1}, {2, 200, 200, 200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 2, 2, 2, 1}, {2, 2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
