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

#include "../../../op_host/arch35/resize_bicubic_v2_grad_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeBicubicV2GradTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeBicubicV2GradTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeBicubicV2GradTilingTest TearDown" << std::endl;
  }
};

TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_01)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData = "64 48 0 32736 ";
    std::vector<size_t> expectWorkspaces = {16777216};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}