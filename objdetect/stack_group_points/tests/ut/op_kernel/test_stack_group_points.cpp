/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;
extern "c" __global__ __aicore__ void stack_group_points(
    GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices, GM_ADDR indices_batch_cnt, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling);

class StackGroupPointsTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "StackGroupPointsTest SetUp\n"
             << endl;
    }
    static void TearDownTestCase()
    {
        cout << "StackGroupPointsTest TearDown\n"
             << endl;
    }
};

struct StackGroupPointsTilingData
{
    int64_t b;
    int64_t m;
    int64_t c;
    int64_t nsample;
    int64_t res;
    int64_t featuresSize;
    int64_t indicesSize;
    int64_t fbcSize;
    int64_t ibcSize;
    int64_t reminder;
    int64_t outLength;
    int64_t n;
    int64_t standard;
    int64_t actCore;
};

TEST_F(StackGroupPointsTest, test_case_fp32)
{
    uint64_t m = 10;
    uint64_t nsample = 6;
    uint64_t n = 5;
    uint64_t c = 12;
    uint64_t b = 2;

    size_t features_bytes_size = 120 * sizeof(float);
    size_t indices_bytes_size = 120 * sizeof(int);
    size_t features_batch_cnt_bytes_size = 120 * sizeof(int);
    size_t indices_batch_cnt_bytes_size = 120 * sizeof(int);
    size_t output_bytes_size = 720 * sizeof(float);
    size_t tiling_data_size = sizeof(StackGroupPointsTilingData);

    uint8_t *features = (uint8_t *)AscendC::GmAlloc(features_bytes_size);
    uint8_t *indices = (uint8_t *)AscendC::GmAlloc(indices_bytes_size);
    uint8_t *features_batch_cnt = (uint8_t *)AscendC::GmAlloc(features_batch_cnt_bytes_size);
    uint8_t *indices_batch_cnt = (uint8_t *)AscendC::GmAlloc(indices_batch_cnt_bytes_size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(output_bytes_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    uint64_t block_dim = 48;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/stack_group_points/stack_group_points_data ./");
    system("chmod -R 755 ./stack_group_points_data/");
    system("cd ./stack_group_points_data/ && rm -rf ./*bin");
    system("cd ./stack_group_points_data/ && python3 gen_data.py 10 6 5 12 2 np.float32");
    system("cd ./stack_group_points_data/ && python3 gen_tiling.py");

    std::string path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(
        stack_group_points, block_dim, features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, tiling);

    AscendC::GmFree((void *)features);
    AscendC::GmFree((void *)features_batch_cnt);
    AscendC::GmFree((void *)indices);
    AscendC::GmFree((void *)indices_batch_cnt);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)workspace);
    free(path_);
}

TEST_F(StackGroupPointsTest, test_case_fp16)
{
    uint64_t m = 10;
    uint64_t nsample = 6;
    uint64_t n = 5;
    uint64_t c = 12;
    uint64_t b = 2;

    size_t features_bytes_size = 120 * sizeof(half);
    size_t indices_bytes_size = 120 * sizeof(int);
    size_t features_batch_cnt_bytes_size = 120 * sizeof(int);
    size_t indices_batch_cnt_bytes_size = 120 * sizeof(int);
    size_t output_bytes_size = 720 * sizeof(half);
    size_t tiling_data_size = sizeof(StackGroupPointsTilingData);

    uint8_t *features = (uint8_t *)AscendC::GmAlloc(features_bytes_size);
    uint8_t *indices = (uint8_t *)AscendC::GmAlloc(indices_bytes_size);
    uint8_t *features_batch_cnt = (uint8_t *)AscendC::GmAlloc(features_batch_cnt_bytes_size);
    uint8_t *indices_batch_cnt = (uint8_t *)AscendC::GmAlloc(indices_batch_cnt_bytes_size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(output_bytes_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    uint64_t block_dim = 48;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/stack_group_points/stack_group_points_data ./");
    system("chmod -R 755 ./stack_group_points_data/");
    system("cd ./stack_group_points_data/ && rm -rf ./*bin");
    system("cd ./stack_group_points_data/ && python3 gen_data.py 10 6 5 12 2 np.float16");
    system("cd ./stack_group_points_data/ && python3 gen_tiling.py");

    std::string path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(0);

    ICPU_RUN_KF(
        stack_group_points, block_dim, features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, tiling);

    AscendC::GmFree((void *)features);
    AscendC::GmFree((void *)features_batch_cnt);
    AscendC::GmFree((void *)indices);
    AscendC::GmFree((void *)indices_batch_cnt);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)workspace);
    free(path_);
}