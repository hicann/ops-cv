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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_linear1d_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class UpsampleLinear1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleLinear1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleLinear1dTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleLinear1dTiling, upsample_linear1d_tiling_001) {
    optiling::UpsampleLinear1dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara("UpsampleLinear1d",
                                              {{{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                               gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2))},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 64 1 1073741824 0 8192 0 0 0 1 64 0 1 1 0 8192 4096 1 0 1 0 64 0 1 0 1 128 1 1 1 128 2 0 1 0 4294967297 549755813888 4294967424 549755813952 274877906960 4294967424 4294967297 1 0 175921860444160 4096 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16810752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleLinear1dTiling, upsample_linear1d_tiling_002) {
    optiling::UpsampleLinear1dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
                                            "UpsampleLinear1d",
                                            {
                                                {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                            },
                                            {
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>({1}))
                                            },
                                            &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 128 2 1065353216 0 0 0 0 0 0 0 0 1 1 0 0 4096 1 0 1 0 0 0 1 0 1 0 1 1 1 128 1 1 1 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777728};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleLinear1dTiling, upsample_linear1d_tiling_003) {
    optiling::UpsampleLinear1dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara("UpsampleLinear1d",
                                            {
                                                {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                                {{{1}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{1, 1, 256}, {1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                            },
                                            {
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>({2}))
                                            },
                                            &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 128 2 1056898815 0 4294976640 2 2 2 1 128 0 1 1 0 2560 4096 1 0 1 0 128 0 1 0 1 80 1 1 1 128 1 1 1 256 4294967297 549755814144 4294967424 313532612736 549755813904 8589934656 4294967298 1 0 316659348799488 8192 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16815424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}