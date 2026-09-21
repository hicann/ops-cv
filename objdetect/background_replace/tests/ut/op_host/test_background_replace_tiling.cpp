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
#include <cstdint>
#include <cstring>
#include <limits>
#include "../../../op_host/background_replace_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class BackgroundReplaceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BackgroundReplaceTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BackgroundReplaceTiling TearDown" << std::endl; }
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
                                              {
                                                  {outShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
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
                                              {
                                                  {outShape, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
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
                                              {
                                                  {outShape, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
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
                                              {
                                                  {outShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "900 ";
    std::vector<size_t> expectWorkspaces = {4294967295};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

static gert::TilingContextPara MakeBackgroundReplaceTilingPara(int64_t bh, int64_t bw, int64_t bc, int64_t mh,
                                                               int64_t mw, int64_t mc, ge::DataType dtype)
{
    static BackgroundReplaceCompileInfo compileInfo = {40, 196608};
    gert::StorageShape bkgShape = {{bh, bw, bc}, {bh, bw, bc}};
    gert::StorageShape srcShape = {{bh, bw, bc}, {bh, bw, bc}};
    gert::StorageShape maskShape = {{mh, mw, mc}, {mh, mw, mc}};
    gert::StorageShape outShape = {{bh, bw, bc}, {bh, bw, bc}};
    return gert::TilingContextPara(
        "BackgroundReplace",
        {{bkgShape, dtype, ge::FORMAT_ND}, {srcShape, dtype, ge::FORMAT_ND}, {maskShape, dtype, ge::FORMAT_ND}},
        {{outShape, dtype, ge::FORMAT_ND}}, {}, &compileInfo);
}

static uint32_t ReadTilingSize(const TilingInfo& info)
{
    uint32_t size = 0;
    EXPECT_GE(info.tilingDataSize, sizeof(uint32_t));
    std::memcpy(&size, info.tilingData.get(), sizeof(uint32_t));
    return size;
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_uint32_max_kept)
{
    auto para = MakeBackgroundReplaceTilingPara(65535, 65537, 1, 65535, 65537, 1, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, 1);
    EXPECT_EQ(ReadTilingSize(info), std::numeric_limits<uint32_t>::max());
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_size_overflow_rejected)
{
    auto para = MakeBackgroundReplaceTilingPara(65536, 65536, 1, 65536, 65536, 1, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_c3_not_wrapped)
{
    auto para = MakeBackgroundReplaceTilingPara(2147483648, 1, 3, 2147483648, 1, 1, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, 3);
    EXPECT_EQ(ReadTilingSize(info), 2147483648U);
}

TEST_F(BackgroundReplaceTiling, background_replace_tiling_size_overflow_offset_rejected)
{
    auto para = MakeBackgroundReplaceTilingPara(65536, 65537, 1, 65536, 65537, 1, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
