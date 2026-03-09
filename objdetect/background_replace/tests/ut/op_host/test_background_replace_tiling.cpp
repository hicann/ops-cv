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
#include <iostream>
#include <vector>
#include "../../../op_host/background_replace_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class BackgroundReplaceTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BackgroundReplaceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BackgroundReplaceTiling TearDown" << std::endl;
    }
};

struct BackgroundReplaceCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(BackgroundReplaceTiling, background_replace_tiling_test_float16_case1)
{
    int h = 20;
    int w = 20;
    int c = 1;

    gert::StorageShape bkgShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape srcShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape maskShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape outShape = {{h, w, c}, {h, w, c}};
    BackgroundReplaceCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("BackgroundReplace",
                                                {{bkgShape, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {srcShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {maskShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                {},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "400 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_test_uint8_case2)
{
    int h = 20;
    int w = 20;
    int c = 1;

    gert::StorageShape bkgShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape srcShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape maskShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape outShape = {{h, w, c}, {h, w, c}};
    BackgroundReplaceCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("BackgroundReplace",
                                                {{bkgShape, ge::DT_UINT8, ge::FORMAT_ND}, 
                                                {srcShape, ge::DT_UINT8, ge::FORMAT_ND},
                                                {maskShape, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{outShape, ge::DT_UINT8, ge::FORMAT_ND},},
                                                {},
                                                &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "400 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_test_uint8_noequal_case3)
{
    int h = 20;
    int w = 20;
    int h2 = 30;
    int w2 = 30;
    int c = 1;

    gert::StorageShape bkgShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape srcShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape maskShape = {{h2, w2, c}, {h2, w2, c}};
    gert::StorageShape outShape = {{h, w, c}, {h, w, c}};
    BackgroundReplaceCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("BackgroundReplace",
                                                {{bkgShape, ge::DT_UINT8, ge::FORMAT_ND}, 
                                                {srcShape, ge::DT_UINT8, ge::FORMAT_ND},
                                                {maskShape, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{outShape, ge::DT_UINT8, ge::FORMAT_ND},},
                                                {},
                                                &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "900 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_test_fp16_noequal_case3)
{
    int h = 20;
    int w = 20;
    int h2 = 30;
    int w2 = 30;
    int c = 1;

    gert::StorageShape bkgShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape srcShape = {{h, w, c}, {h, w, c}};
    gert::StorageShape maskShape = {{h2, w2, c}, {h2, w2, c}};
    gert::StorageShape outShape = {{h, w, c}, {h, w, c}};
    BackgroundReplaceCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("BackgroundReplace",
                                                {{bkgShape, ge::DT_FLOAT16, ge::FORMAT_ND}, 
                                                {srcShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {maskShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                {},
                                                &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "900 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}