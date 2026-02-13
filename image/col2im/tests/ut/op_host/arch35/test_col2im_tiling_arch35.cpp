/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/col2im_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class Col2imTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Col2imTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Col2imTiling TearDown" << std::endl;
    }
};

struct Col2imCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(Col2imTiling, col2im_tiling_test_float32_case1)
{
    int n = 8;
    int c = 64;
    int h_col = 22;
    int w_col = 1;
    int w_k = 5;
    int h_k = 1;
    int h = 20;
    int w = 21;

    gert::StorageShape gradOutShape = {{n, c, w_k*h_k, w_col*h_col}, {n, c, w_k*h_k, w_col*h_col}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape gradInShape = {{n, c, h, w}, {n, c, h, w}};
    std::vector<int32_t> inputSizeValues = {h, w};
    Col2imCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("Col2im",
                                                {{gradOutShape, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},},
                                                {{gradInShape, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                {gert::TilingContextPara::OpAttr("kernel_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 5})),
                                                       gert::TilingContextPara::OpAttr("dilation", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 7})),
                                                       gert::TilingContextPara::OpAttr("padding", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 5})),
                                                       gert::TilingContextPara::OpAttr("stride", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 7}))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "215040 20 21 1 5 2 7 1 5 1 7 22 1 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}