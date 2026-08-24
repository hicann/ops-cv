/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/stack_group_points_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class StackGroupPointsTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StackGroupPointsTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StackGroupPointsTiling TearDown" << std::endl; }
};

static constexpr uint64_t SOC_CORE_NUM = 64;
static constexpr uint64_t SOC_UB_SIZE = 262144;
static constexpr uint64_t TILING_DATA_SIZE = 4096;

TEST_F(StackGroupPointsTiling, stack_group_points_tiling_fp16)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    uint64_t expectTilingKey = 1;
    std::string expectTilingData = "16 32 128 1 4096 65536 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StackGroupPointsTiling, stack_group_points_tiling_fp32)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "16 32 128 1 4096 65536 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 边界校验失败用例 =====

// Negative: features C=0, kernel 中 index / nsample % c 会除零
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_c_zero)
{
    gert::StorageShape featuresShape = {{4096, 0}, {4096, 0}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 0, 128}, {16, 0, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices nsample=0, kernel 中 index % nsample 会除零
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_nsample_zero)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 0}, {16, 0}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 0}, {16, 32, 0}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt B=0
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_batch_cnt_b_zero)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{0}, {0}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{0}, {0}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt dim0 != indices_batch_cnt dim0
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_fbc_b_mismatch)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{2}, {2}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{3}, {3}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features dtype 不支持（DT_DOUBLE）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_dtype_double)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features 非 2D（3D）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_not_2d)
{
    gert::StorageShape featuresShape = {{4096, 32, 8}, {4096, 32, 8}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
