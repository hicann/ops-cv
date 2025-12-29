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

#include "../../../op_host/arch35/resize_bicubic_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeBicubicV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ResizeBicubicV2TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ResizeBicubicV2TilingTest TearDown" << std::endl;
    }
};

TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_01)
{
    gert::StorageShape inputXShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape inputSizeShape = {{1, 2}, {1, 2}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 65541;
    string expectTilingData = "64 48 0 32 32 32 32 3 1 4575657222473777152 1 32 32 3 1 32 32 25056 ";
    std::vector<size_t> expectWorkspaces = {16777216};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}