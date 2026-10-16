/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include "../../../op_host/arch22/stack_group_points_tiling.h"

using namespace std;
extern "C" __global__ __aicore__ void stack_group_points(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices,
                                                         GM_ADDR indices_batch_cnt, GM_ADDR y, GM_ADDR workspace,
                                                         GM_ADDR tiling);

class StackGroupPointsTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "StackGroupPointsTest SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "StackGroupPointsTest TearDown\n" << endl; }
};

TEST_F(StackGroupPointsTest, test_case_fp32)
{
    uint64_t m = 10;
    uint64_t nsample = 6;
    uint64_t n = 5;
    uint64_t c = 12;
    uint64_t b = 2;

    size_t features_bytes_size = n * c * sizeof(float);
    size_t features_batch_cnt_bytes_size = (32) * sizeof(int);
    size_t indices_bytes_size = m * nsample * sizeof(int);
    size_t indices_batch_cnt_bytes_size = 32 * sizeof(int);
    size_t output_bytes_size = (m * c * nsample) * sizeof(float);
    size_t tiling_data_size = sizeof(StackGroupPointsTilingData);

    uint8_t* features = (uint8_t*)AscendC::GmAlloc(features_bytes_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_bytes_size);
    uint8_t* features_batch_cnt = (uint8_t*)AscendC::GmAlloc(features_batch_cnt_bytes_size);
    uint8_t* indices_batch_cnt = (uint8_t*)AscendC::GmAlloc(indices_batch_cnt_bytes_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(output_bytes_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    uint64_t block_dim = 48;

    system("cp -r ../../../../objdetect/stack_group_points/tests/ut/op_kernel/stack_group_points_data ./");
    system("chmod -R 755 ./stack_group_points_data/");
    system("cd ./stack_group_points_data/ && rm -rf ./*bin");
    system("cd ./stack_group_points_data/ && python3 gen_data.py 10 6 5 12 2 np.float32");
    system("cd ./stack_group_points_data/ && python3 gen_tiling.py np.float32");

    std::string path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/stack_group_points_data/features.bin", features_bytes_size, features, features_bytes_size);
    ReadFile(path + "/stack_group_points_data/indices.bin", indices_bytes_size, indices, indices_bytes_size);
    ReadFile(path + "/stack_group_points_data/features_batch_cnt.bin", features_batch_cnt_bytes_size,
             features_batch_cnt, features_batch_cnt_bytes_size);
    ReadFile(path + "/stack_group_points_data/indices_batch_cnt.bin", indices_batch_cnt_bytes_size, indices_batch_cnt,
             indices_batch_cnt_bytes_size);
    ReadFile(path + "/stack_group_points_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(stack_group_points, block_dim, features, features_batch_cnt, indices, indices_batch_cnt, y, workspace,
                tiling);

    AscendC::GmFree((void*)features);
    AscendC::GmFree((void*)features_batch_cnt);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)indices_batch_cnt);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)workspace);
}

TEST_F(StackGroupPointsTest, test_case_fp32_batch_local_index_bounds)
{
    constexpr size_t kInputBytes = 32;
    constexpr size_t kOutputElements = 16;
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* features_batch_cnt = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* indices_batch_cnt = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(kOutputElements * sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(StackGroupPointsTilingData));

    ASSERT_NE(features, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(features_batch_cnt, nullptr);
    ASSERT_NE(indices_batch_cnt, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memset(features, 0, kInputBytes);
    std::memset(indices, 0, kInputBytes);
    std::memset(features_batch_cnt, 0, kInputBytes);
    std::memset(indices_batch_cnt, 0, kInputBytes);
    std::fill_n(reinterpret_cast<float*>(y), kOutputElements, 0.0f);
    const std::array<float, 6> featureValues = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<int32_t, 3> indexValues = {0, 2, 1};
    const std::array<int32_t, 3> featureBatchCounts = {2, 2, 2};
    const std::array<int32_t, 3> indexBatchCounts = {1, 1, 1};
    std::memcpy(features, featureValues.data(), sizeof(featureValues));
    std::memcpy(indices, indexValues.data(), sizeof(indexValues));
    std::memcpy(features_batch_cnt, featureBatchCounts.data(), sizeof(featureBatchCounts));
    std::memcpy(indices_batch_cnt, indexBatchCounts.data(), sizeof(indexBatchCounts));

    // Tiling fields follow StackGroupPointsTilingData serialization order.
    const std::array<int64_t, 14> tilingValues = {3, 3, 1, 1, 0, 32, 32, 32, 32, 1, 32, 6, 3, 48};
    std::memcpy(tiling, tilingValues.data(), sizeof(tilingValues));

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(stack_group_points, 1, features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, tiling);

    const float* output = reinterpret_cast<const float*>(y);
    EXPECT_FLOAT_EQ(output[0], 10.0f);
    EXPECT_FLOAT_EQ(output[1], 0.0f);
    EXPECT_FLOAT_EQ(output[2], 31.0f);

    AscendC::GmFree((void*)features);
    AscendC::GmFree((void*)features_batch_cnt);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)indices_batch_cnt);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)workspace);
}

TEST_F(StackGroupPointsTest, test_case_fp32_output_tail_guard)
{
    constexpr size_t kInputBytes = 32;
    constexpr size_t kOutputElements = 16;
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* features_batch_cnt = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* indices_batch_cnt = (uint8_t*)AscendC::GmAlloc(kInputBytes);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(kOutputElements * sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(StackGroupPointsTilingData));

    ASSERT_NE(features, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(features_batch_cnt, nullptr);
    ASSERT_NE(indices_batch_cnt, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memset(features, 0, kInputBytes);
    std::memset(indices, 0, kInputBytes);
    std::memset(features_batch_cnt, 0, kInputBytes);
    std::memset(indices_batch_cnt, 0, kInputBytes);
    std::fill_n(reinterpret_cast<float*>(y), kOutputElements, 12345.0f);
    *reinterpret_cast<float*>(features) = 7.0f;
    *reinterpret_cast<int32_t*>(features_batch_cnt) = 1;
    *reinterpret_cast<int32_t*>(indices_batch_cnt) = 1;

    // Tiling fields follow StackGroupPointsTilingData serialization order.
    const std::array<int64_t, 14> tilingValues = {1, 1, 1, 1, 0, 32, 32, 32, 32, 1, 32, 1, 1, 48};
    std::memcpy(tiling, tilingValues.data(), sizeof(tilingValues));

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(stack_group_points, 1, features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, tiling);

    const float* output = reinterpret_cast<const float*>(y);
    EXPECT_FLOAT_EQ(output[0], 7.0f);
    EXPECT_FLOAT_EQ(output[1], 12345.0f);

    AscendC::GmFree((void*)features);
    AscendC::GmFree((void*)features_batch_cnt);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)indices_batch_cnt);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)workspace);
}
