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
#include "../../../op_host/grid_sample_tiling.h"
#include "data_utils.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

extern "C" void grid_sample(uint8_t *x, uint8_t *grid, uint8_t *y, uint8_t *workspace, uint8_t *tiling);

class grid_sample_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "grid_sample_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "grid_sample_test TearDown\n" << endl;
    }
};

TEST_F(grid_sample_test, test_case_fp32_test01)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 2, 2, 1}, {2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test02)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test03)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test04)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 8}, {2, 64, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test05)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test06)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test07)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp16_test08)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp16_test09)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp16_test10)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_fp32_test11)
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
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bicubic_fp16_yunzhuduan)
{
    int64_t N = 1;
    int64_t x_h = 608;
    int64_t x_w = 608;
    int64_t C = 3;
    int64_t grid_h = 832;
    int64_t grid_w = 832;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{1, 608, 608, 3}, {1, 608, 608, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 832, 832, 2}, {1, 832, 832, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{1, 3, 832, 832}, {1, 3, 832, 832}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bicubic_fp16_codetr)
{
    int64_t N = 8;
    int64_t x_h = 24;
    int64_t x_w = 24;
    int64_t C = 32;
    int64_t grid_h = 900;
    int64_t grid_w = 4;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{8, 32, 24, 24}, {8, 32, 24, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 900, 4, 2}, {8, 900, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 32, 900, 4}, {8, 32, 900, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bicubic_fp32_yunzhuduan)
{
    int64_t N = 1;
    int64_t x_h = 608;
    int64_t x_w = 608;
    int64_t C = 3;
    int64_t grid_h = 832;
    int64_t grid_w = 832;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{1, 608, 608, 3}, {1, 608, 608, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 832, 832, 2}, {1, 832, 832, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 3, 832, 832}, {1, 3, 832, 832}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bicubic_fp16_fullLoad)
{
    int64_t N = 8;
    int64_t x_h = 24;
    int64_t x_w = 24;
    int64_t C = 32;
    int64_t grid_h = 900;
    int64_t grid_w = 4;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{8, 32, 24, 24}, {8, 32, 24, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 900, 4, 2}, {8, 900, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 32, 900, 4}, {8, 32, 900, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bicubic_fp32_fullLoad)
{
    int64_t N = 1;
    int64_t x_h = 608;
    int64_t x_w = 608;
    int64_t C = 1;
    int64_t grid_h = 832;
    int64_t grid_w = 832;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{1, 608, 608, 1}, {1, 608, 608, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 832, 832, 2}, {1, 832, 832, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 1, 832, 832}, {1, 1, 832, 832}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bicubic")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp16_fullLoad_c1)
{
    int64_t N = 8;
    int64_t x_h = 24;
    int64_t x_w = 24;
    int64_t C = 1;
    int64_t grid_h = 900;
    int64_t grid_w = 4;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{8, 24, 24, 1}, {8, 24, 24, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 900, 4, 2}, {8, 900, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 1, 900, 4}, {8, 1, 900, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp32_fullLoad_c1)
{
    int64_t N = 1;
    int64_t x_h = 608;
    int64_t x_w = 608;
    int64_t C = 1;
    int64_t grid_h = 832;
    int64_t grid_w = 832;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{1, 608, 608, 1}, {1, 608, 608, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 832, 832, 2}, {1, 832, 832, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 1, 832, 832}, {1, 1, 832, 832}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp16_fullLoad_c32)
{
    int64_t N = 8;
    int64_t x_h = 24;
    int64_t x_w = 24;
    int64_t C = 32;
    int64_t grid_h = 900;
    int64_t grid_w = 4;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{8, 32, 24, 24}, {8, 32, 24, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 900, 4, 2}, {8, 900, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 32, 900, 4}, {8, 32, 900, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp32_fullLoad_c32)
{
    int64_t N = 1;
    int64_t x_h = 608;
    int64_t x_w = 608;
    int64_t C = 32;
    int64_t grid_h = 832;
    int64_t grid_w = 832;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{1, 608, 608, 32}, {1, 608, 608, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 832, 832, 2}, {1, 832, 832, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 32, 832, 832}, {1, 32, 832, 832}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp32_test01)
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
    size_t inputByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 2, 2, 2, 1}, {2, 2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_bilinear_fp16_test02)
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
    size_t inputByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(grid_sample_test, test_case_nearest_fp32_test03)
{
    cout << "test_case_nearest_fp32_test03 start\n" << endl;
    int64_t N = 2;
    int64_t x_d = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    size_t inputByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 2, 2, 2, 1}, {2, 2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);
    cout << "ICPU_RUN_KF end\n" << endl;

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
TEST_F(grid_sample_test, test_case_nearest_fp16_test04)
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
    size_t inputByteSize = N * x_d * x_h * x_w * C * sizeof(int32_t);
    size_t gridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    size_t outputByteSize = N * grid_d * grid_h * grid_w * C * sizeof(int32_t);
    struct GridSampleCompileInfo {
        int64_t coreNum = 48;
    }compileInfo;
    gert::TilingContextPara tilingContextPara("GridSample",
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 1, 2, 2, 2}, {2, 1, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
                                                gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
                                                &compileInfo, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *grid = (uint8_t *)AscendC::GmAlloc(gridByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char *path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sample, blockDim, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}