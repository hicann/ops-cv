/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/check_valid_tiling_data.h"

using namespace std;
using namespace ge;

class CheckValidTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckValidTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckValidTiling TearDown" << std::endl; }
};

std::map<std::string, std::string> soc_versions_infos = {{"Short_SoC_version", "Ascend950"}};

constexpr size_t CV_SYS_WORKSPACE_SIZE = 16777216;

TEST_F(CheckValidTiling, cv_fp32_normal)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 5456 8 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp16_normal)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 10912 8 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp32_empty)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 1}, {0, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "0 0 0 0 0 0 0 0 -4647714812233515008 ",
                    {0});
}

TEST_F(CheckValidTiling, cv_fp16_empty)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{0, 4}, {0, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 1}, {0, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "0 0 0 0 0 0 0 0 -4647714812233515008 ",
                    {0});
}

TEST_F(CheckValidTiling, cv_fp32_large_multicore)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{100000, 4}, {100000, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{100000, 1}, {100000, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "100000 5456 1792 19 19 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp16_nonaligned)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{33, 4}, {33, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{33, 1}, {33, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "33 10912 33 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_invalid_rank1)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_last_dim)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_dtype_int64)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_dtype_mismatch)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_img_metas_short)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
