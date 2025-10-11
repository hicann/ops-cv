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
#include <cstdint>
#include <iostream>
#include <string>

#include <vector>

#include "data_utils.h"
#include "grid_sampler2_d_grad.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_data_def.h"

extern "C" __global__ __aicore__ void grid_sampler2_d_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx, GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling);

class grid_sampler_2d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "grid_sampler_2d_grad SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grid_sampler_2d_grad TearDown\n" << std::endl;
    }
};

TEST_F(grid_sampler_2d_grad_test, test_float_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t grad_size = N * C * gH * gW * sizeof(float);
    size_t grid_size = N * 2 * gH * gW * sizeof(float);
    size_t dx_size = N * C * H * W * sizeof(float);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 0, 0, 1};

    ReadFile("./gen_data/tiling.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 4>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_float1_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t grad_size = N * C * gH * gW * sizeof(float);
    size_t grid_size = N * 2 * gH * gW * sizeof(float);
    size_t dx_size = N * C * H * W * sizeof(float);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 1, 0, 1};

    ReadFile("./gen_data/tiling.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 4>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_fp16.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 8;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t grad_size = N * C * gH * gW * sizeof(half);
    size_t grid_size = N * 2 * gH * gW * sizeof(half);
    size_t dx_size = N * C * H * W * sizeof(half);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 0, 0, 1};

    ReadFile("./gen_data/tiling_16.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x_16.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad_16.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid_16.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx_16.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid_16.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 2>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half1_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_fp16.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t grad_size = N * C * gH * gW * sizeof(half);
    size_t grid_size = N * 2 * gH * gW * sizeof(half);
    size_t dx_size = N * C * H * W * sizeof(half);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 1, 0, 1};

    ReadFile("./gen_data/tiling_16.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x_16.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad_16.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid_16.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx_16.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid_16.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 2>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half2_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_bf16.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t grad_size = N * C * gH * gW * sizeof(half);
    size_t grid_size = N * 2 * gH * gW * sizeof(half);
    size_t dx_size = N * C * H * W * sizeof(half);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 0, 0, 1};

    ReadFile("./gen_data/tiling_16.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x_16.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad_16.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid_16.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx_16.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid_16.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 2>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half3_case)
{
    system(
        "cp -rf "
        "../../../../image/grid_sampler2_d_grad/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_bf16.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 2 * 1024;
    int32_t op_code = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(GridSampler2DGradTilingDataTest);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t grad_size = N * C * gH * gW * sizeof(half);
    size_t grid_size = N * 2 * gH * gW * sizeof(half);
    size_t dx_size = N * C * H * W * sizeof(half);
    size_t dgrid_size = N * 2 * gH * gW * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(grid_size);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgrid_size);

    struct InputParamsInfo params = {N, C, H, W, gH, gW, 1, 0, 1};

    ReadFile("./gen_data/tiling_16.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x_16.bin", x_size, x, x_size);
    ReadFile("./gen_data/grad_16.bin", grad_size, grad, grad_size);
    ReadFile("./gen_data/grid_16.bin", grid_size, grid, grid_size);

    WriteFile("./gen_data/dx_16.bin", dx, dx_size);
    WriteFile("./gen_data/dgrid_16.bin", dgrid, dgrid_size);

    optiling::GetGridSampler2DGradTiling<GridSampler2DGradTilingDataTest, 2>(
        reinterpret_cast<GridSampler2DGradTilingDataTest*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(
        grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace,
        tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)grid);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgrid);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
