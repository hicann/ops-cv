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
#include "../../../op_host/grid_sampler_2d_grad_tiling_data.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace optiling;

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
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_float1_case)
{
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half_case)
{
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half1_case)
{
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half2_case)
{
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(grid_sampler_2d_grad_test, test_half3_case)
{
    uint32_t N = 8;
    uint32_t C = 8;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t gH = 8;
    uint32_t gW = 8;
    int64_t dim = 2;
    size_t gradByteSize = N * C * gH * gW * sizeof(float);
    size_t xByteSize = N * C * H * W * sizeof(float);
    size_t gridByteSize = N * 2 * gH * gW * sizeof(float);
    size_t dxByteSize = N * C * H * W * sizeof(float);
    size_t dgridByteSize = N * 2 * gH * gW * sizeof(float);
    Tiling4GridSampler2DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND}},
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 8, 8, 2}, {8, 8, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgrid = (uint8_t*)AscendC::GmAlloc(dgridByteSize);

    uint32_t blockDim = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(grid_sampler2_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
