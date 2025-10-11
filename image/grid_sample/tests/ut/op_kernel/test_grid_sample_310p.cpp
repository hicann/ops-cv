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
#include "data_utils.h"
#include "test_grid_sample_310p.h"

#include <cstdint>

using namespace std;

extern "C" void grid_sample(uint8_t *x, uint8_t *grid, uint8_t *y, uint8_t *workspace, uint8_t *tiling);

class grid_sample_310p_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "grid_sample_310p_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "grid_sample_310p_test TearDown\n" << endl;
    }
};

TEST_F(grid_sample_310p_test, test_case_fp32_test08)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampleTilingDataTestP);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    GridSampleTilingDataTestP *tilingDatafromBin = reinterpret_cast<GridSampleTilingDataTestP *>(tiling);
    tilingDatafromBin->coreNumVar = 48;
    tilingDatafromBin->inN = 2;
    tilingDatafromBin->inC = 1;
    tilingDatafromBin->inH = 2;
    tilingDatafromBin->inW = 2;
    tilingDatafromBin->outH = 2;
    tilingDatafromBin->outW = 2;
    tilingDatafromBin->interpolationMode = 0;
    tilingDatafromBin->paddingMode = 0;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->channelLast = 1;
    tilingDatafromBin->needCoreNum = 2;
    tilingDatafromBin->preCoreNum = 0;
    tilingDatafromBin->preNumPerCore = 0;
    tilingDatafromBin->postNumPerCore = 0;

    ICPU_SET_TILING_KEY(2001220);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_310p_test, test_case_fp32_test11)
{
    int64_t N = 2;
    int64_t x_h = 16;
    int64_t x_w = 16;
    int64_t C = 64;
    int64_t grid_h = 8;
    int64_t grid_w = 64;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GridSampleTilingDataTestP);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    GridSampleTilingDataTestP *tilingDatafromBin = reinterpret_cast<GridSampleTilingDataTestP *>(tiling);
    tilingDatafromBin->coreNumVar = 48;
    tilingDatafromBin->inN = 2;
    tilingDatafromBin->inC = 64;
    tilingDatafromBin->inH = 16;
    tilingDatafromBin->inW = 16;
    tilingDatafromBin->outH = 8;
    tilingDatafromBin->outW = 64;
    tilingDatafromBin->interpolationMode = 1;
    tilingDatafromBin->paddingMode = 1;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->channelLast = 1;
    tilingDatafromBin->needCoreNum = 48;
    tilingDatafromBin->preCoreNum = 0;
    tilingDatafromBin->preNumPerCore = 0;
    tilingDatafromBin->postNumPerCore = 0;

    ICPU_SET_TILING_KEY(1001220);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}