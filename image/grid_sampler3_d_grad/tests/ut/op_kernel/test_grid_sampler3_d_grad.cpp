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
#include "../../../op_host/grid_sampler3_d_grad_tiling.h"
#include "data_utils.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 1}, {2, 1, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 1, 1, 2, 2}, {2, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 800}, {2, 1, 2, 2, 800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 800}, {2, 1, 2, 2, 800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 800, 1, 2, 2}, {2, 800, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

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
    size_t dgridByteSize = N * grid_d * grid_h * grid_w * dim * sizeof(int32_t);
    Tiling4GridSampler3DGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{{{2, 1, 2, 2, 500}, {2, 1, 2, 2, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 1, 2, 2, 500}, {2, 1, 2, 2, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 500, 1, 2, 2}, {2, 500, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("reflection")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
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

    ICPU_RUN_KF(grid_sampler3_d_grad, blockDim, grad, x, grid, dx, dgrid, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgrid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}