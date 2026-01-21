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

#include "../../../op_host/upsample_nearest3d_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace optiling;

class UpsampleNearest3dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleNearest3dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleNearest3dTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleNearest3dTiling, Ascend910B_upsample_nearest3d_tiling_001)
{
    UpsampleNearest3dCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("UpsampleNearest3d",
        {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
        gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 5140;
    string expectTilingData = "0 1 128 128 128 256 256 256 4539628425446424576 1056964608 2048 1028 4 4 128 128 341 16 16368 1 1 1 48 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, Ascend310P_upsample_nearest3d_tiling_002)
{
    UpsampleNearest3dCompileInfo compileInfo = {48};
    string socVersion = "Ascend310p";
    gert::TilingContextPara tilingContextPara("UpsampleNearest3d",
        {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
        gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 5140;
    string expectTilingData = "0 1 128 128 128 256 256 256 4539628425446424576 1056964608 2048 1028 4 4 128 128 341 16 16368 1 1 1 48 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}