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

#include "../../../op_host/arch35/resize_linear_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeLinearTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeLinearTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeLinearTilingTest TearDown" << std::endl;
  }
};

TEST_F(ResizeLinearTilingTest, resize_linear_tiling_01)
{
    gert::StorageShape inputXShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputSizeShape = {{1,}, {1,}};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};
    int size_value[1] = {32};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 65793;
    string expectTilingData = "64 1 32 32 32 1065353216 ";
    std::vector<size_t> expectWorkspaces = {16777216};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}