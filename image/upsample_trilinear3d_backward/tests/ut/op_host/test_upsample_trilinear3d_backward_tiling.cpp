/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/upsample_trilinear3d_backward_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace gert;
using namespace optiling;

class UpsampleTrilinear3dBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleTrilinear3dBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleTrilinear3dBackwardTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleTrilinear3dBackwardTiling, upsample_trilinear3d_tiling_001)
{
    UpsampleTrilinearBackwardCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("UpsampleTrilinear3dBackward",
        {{{{1, 1, 4, 128, 128}, {1, 1, 4, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 12, 256, 256}, {1, 1, 12, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr(
             "output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({12, 256, 256})),
            gert::TilingContextPara::OpAttr(
                "input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 4, 128, 128})),
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
            gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
            gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "2 1 12 256 256 4 128 128 4611686019501129728 72340169604202496 16 224 131072 262144 0 0 0 1 16 16 0 0 0 1 1 3 "
        "1 4 512 16 16 171 1 16 48 2199023255553 549755814144 2199023255680 60129542160 68719477248 4294967312 "
        "4294967297 1 0 145135534866432 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "1099511627777 549755814144 68719476864 60129542400 1099511627792 4294967312 4294967297 1 0 74766790688768 "
        "16384 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 51539607553 17179934720 68719476740 "
        "17179934720 1099511627792 4294967304 4294967312 8 4294967296 565148976676864 16384 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {35170304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
