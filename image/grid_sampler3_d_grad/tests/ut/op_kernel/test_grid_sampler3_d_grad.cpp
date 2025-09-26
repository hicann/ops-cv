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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "impl/dav_c220/kfc/kfc_log.h"
#include "test_grid_sampler3_d_grad.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void grid_sampler3_d_grad(
    uint8_t* grad, uint8_t* x, uint8_t* grid, uint8_t* dx, uint8_t* dgrid, uint8_t* workspace, uint8_t* tiling);

class grid_sampler3_d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "grid_sampler3_d_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "grid_sampler3_d_grad_test TearDown\n" << endl;
    }
};

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test01)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 0;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test02)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 1;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test03)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 2;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test04)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 1;
    tilingDatafromBin->padding = 0;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test05)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 1;
    tilingDatafromBin->padding = 1;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test06)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 1;
    tilingDatafromBin->padding = 2;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test07)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 0;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test08)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 1;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test09)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 2;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test10)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 1;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 0;
    tilingDatafromBin->padding = 0;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test11)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 800;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 800;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 1;
    tilingDatafromBin->padding = 1;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sampler3_d_grad_test, test_case_fp32_test12)
{
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 500;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t gradByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t xByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t dxByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampler3DGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    GridSampler3DGradTilingData* tilingDatafromBin = reinterpret_cast<GridSampler3DGradTilingData*>(tiling);
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->channel = 500;
    tilingDatafromBin->xD = 1;
    tilingDatafromBin->xH = 2;
    tilingDatafromBin->xW = 2;
    tilingDatafromBin->gridD = 2;
    tilingDatafromBin->gridH = 2;
    tilingDatafromBin->gridW = 2;
    tilingDatafromBin->interpolation = 1;
    tilingDatafromBin->padding = 2;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->blockNum = 16;
    tilingDatafromBin->pNumPerCore = 1;
    tilingDatafromBin->dxNumPerCore = 1;
    tilingDatafromBin->tailPNum = 0;
    tilingDatafromBin->group = 1;
    tilingDatafromBin->ubFactorElement = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}