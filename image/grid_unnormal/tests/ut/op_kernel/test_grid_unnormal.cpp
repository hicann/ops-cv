/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grid_unnormal.cpp
 * \brief Kernel UT for grid_unnormal operator (CPU 仿真 / RegBase)
 *
 * 按 dtype 实例化 kernel，在 CPU 仿真下以单核执行并校验 diff / position 输出。
 */

#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "grid_unnormal_tiling.h"

using namespace std;

#undef __CCE_KT_TEST__
#define DTYPE_GRID float
#define grid_unnormal grid_unnormal_float_entry
#include "../../../op_kernel/grid_unnormal.cpp"
#undef grid_unnormal
#undef DTYPE_GRID
#define DTYPE_GRID half
#define grid_unnormal grid_unnormal_half_entry
#include "../../../op_kernel/grid_unnormal.cpp"
#undef grid_unnormal
#undef DTYPE_GRID
#define __CCE_KT_TEST__

class GridUnnormalKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "GridUnnormalKernelTest SetUp" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(std::system(cmd.c_str()), 0);
        ASSERT_EQ(std::system("chmod -R 755 ./grid_unnormal_data/"), 0);
    }
    static void TearDownTestCase() { cout << "GridUnnormalKernelTest TearDown" << endl; }

private:
    const static std::string dataPath;
};

const std::string GridUnnormalKernelTest::dataPath = std::string(GRID_UNNORMAL_UT_DATA_DIR);

namespace {
void FillTiling(GridUnnormalTilingData* td, int64_t total, int64_t blockDim, int32_t alignCorners)
{
    td->totalNum = total;
    td->perCoreNum = (total == 0) ? 0 : (total / blockDim + (total % blockDim != 0));
    td->ubFactor = 64; // 64 元素 tile，覆盖多轮与尾块场景
    td->alignCorners = alignCorners;
}

template <typename T>
void RunKernel(int64_t total, int64_t blockDim, int32_t alignCorners, size_t tByte, size_t posByte,
               const std::string& gridBin, const std::string& assistBin, const std::string& diffOut,
               const std::string& posOut)
{
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(tByte);
    uint8_t* assist = (uint8_t*)AscendC::GmAlloc(tByte);
    ReadFile(gridBin, tByte, grid, tByte);
    ReadFile(assistBin, tByte, assist, tByte);
    uint8_t* diff = (uint8_t*)AscendC::GmAlloc(tByte);
    uint8_t* position = (uint8_t*)AscendC::GmAlloc(posByte);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(GridUnnormalTilingData));

    FillTiling(reinterpret_cast<GridUnnormalTilingData*>(tiling), total, blockDim, alignCorners);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if constexpr (std::is_same<T, float>::value) {
        ICPU_RUN_KF(grid_unnormal_float_entry, blockDim, grid, assist, diff, position, workspace, tiling);
    } else {
        ICPU_RUN_KF(grid_unnormal_half_entry, blockDim, grid, assist, diff, position, workspace, tiling);
    }

    WriteFile(diffOut, diff, tByte);
    WriteFile(posOut, position, posByte);

    AscendC::GmFree(grid);
    AscendC::GmFree(assist);
    AscendC::GmFree(diff);
    AscendC::GmFree(position);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

// fp32, align_corners=false, shape (5,5,4,2)=200（>ubFactor → 多 tile + 尾块）
TEST_F(GridUnnormalKernelTest, test_case_fp32_align_false)
{
    const int64_t total = 5 * 5 * 4 * 2;
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 gen_data.py '(5,5,4,2)' 'float32' 'False'"), 0);
    RunKernel<float>(total, 1, 0, total * sizeof(float), total * sizeof(int32_t),
                     "./grid_unnormal_data/float32_grid_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_assist_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_diff_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_position_grid_unnormal.bin");
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 compare_data.py 'float32'"), 0);
}

// fp16, align_corners=true, shape (3,4,5,2)=120
TEST_F(GridUnnormalKernelTest, test_case_fp16_align_true)
{
    const int64_t total = 3 * 4 * 5 * 2;
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 gen_data.py '(3,4,5,2)' 'float16' 'True'"), 0);
    RunKernel<half>(total, 1, 1, total * sizeof(uint16_t), total * sizeof(int32_t),
                    "./grid_unnormal_data/float16_grid_grid_unnormal.bin",
                    "./grid_unnormal_data/float16_assist_grid_unnormal.bin",
                    "./grid_unnormal_data/float16_output_diff_grid_unnormal.bin",
                    "./grid_unnormal_data/float16_output_position_grid_unnormal.bin");
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 compare_data.py 'float16'"), 0);
}

TEST_F(GridUnnormalKernelTest, test_case_fp32_multi_core)
{
    const int64_t total = 5 * 5 * 4 * 2;
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 gen_data.py '(5,5,4,2)' 'float32' 'False'"), 0);
    RunKernel<float>(total, 4, 0, total * sizeof(float), total * sizeof(int32_t),
                     "./grid_unnormal_data/float32_grid_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_assist_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_diff_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_position_grid_unnormal.bin");
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 compare_data.py 'float32'"), 0);
}

TEST_F(GridUnnormalKernelTest, test_case_empty_tensor)
{
    const int64_t total = 0;
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 gen_data.py '(0,6,5,2)' 'float32' 'False'"), 0);
    RunKernel<float>(total, 1, 0, 0, 0, "./grid_unnormal_data/float32_grid_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_assist_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_diff_grid_unnormal.bin",
                     "./grid_unnormal_data/float32_output_position_grid_unnormal.bin");
    ASSERT_EQ(std::system("cd ./grid_unnormal_data/ && python3 compare_data.py 'float32'"), 0);
}
