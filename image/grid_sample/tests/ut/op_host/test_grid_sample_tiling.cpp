/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include "op_log.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "image/grid_sample/op_host/grid_sample_tiling.h"
#include "image_ops.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"

class GridSampleTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSampleTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSampleTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

void TilingTest(std::initializer_list<int64_t> &xShape, std::initializer_list<int64_t> &gridShape,
    std::initializer_list<int64_t> &outShape, ge::DataType datatype, string interpolation_mode, string padding_mode,
    bool align_corners, bool channel_last, int64_t scheduler_mode, const ge::graphStatus status,
    uint64_t tilingKeyValue, string expectData)
{
    std::string op_type("GridSample");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 48
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GridSampleCompileInfo compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());

    gert::StorageShape x = {xShape, xShape};
    gert::StorageShape grid = {gridShape, gridShape};
    gert::StorageShape output = {outShape, outShape};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &grid})
                      .OutputShapes({&output})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, datatype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, datatype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, datatype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>(interpolation_mode)},
                          {"padding_mode", ge::AnyValue::CreateFrom<std::string>(padding_mode)},
                          {"align_corners", ge::AnyValue::CreateFrom<bool>(align_corners)},
                          {"channel_last", ge::AnyValue::CreateFrom<bool>(channel_last)},
                          {"scheduler_mode", ge::AnyValue::CreateFrom<bool>(scheduler_mode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, tilingKeyValue);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_data_result, expectData);
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_1)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "zeros",
        true,
        true,
        0,
        status,
        1000220,
        "48 2 1 0 200 200 0 2 2 0 0 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_2)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "border",
        true,
        true,
        0,
        status,
        1000220,
        "48 2 1 0 200 200 0 2 2 0 1 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_3)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "reflection",
        true,
        true,
        0,
        status,
        1000220,
        "48 2 1 0 200 200 0 2 2 0 2 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float16)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT16,
        "bilinear",
        "reflection",
        true,
        true,
        0,
        status,
        1000210,
        "48 2 1 0 200 200 0 2 2 0 2 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_nearest)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "nearest",
        "reflection",
        true,
        true,
        0,
        status,
        1000221,
        "48 2 1 0 2 2 0 2 2 1 2 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_float32_bicubic)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bicubic",
        "reflection",
        true,
        true,
        0,
        status,
        1000222,
        "48 2 1 0 2 2 0 2 2 2 2 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_slicing_window)
{
    int64_t N = 2;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "border",
        true,
        true,
        1,
        status,
        1001220,
        "48 2 1 0 200 200 0 2 2 0 1 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c2)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "border",
        true,
        true,
        1,
        status,
        2001220,
        "48 2 2 0 2 2 0 2 2 0 1 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c1)
{
    int64_t N = 2;
    int64_t x_h = 2;
    int64_t x_w = 2;
    int64_t C = 1;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "border",
        true,
        true,
        1,
        status,
        2101220,
        "48 2 1 0 2 2 0 2 2 0 1 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_tiling_test_full_load_c32)
{
    int64_t N = 2;
    int64_t x_h = 10;
    int64_t x_w = 10;
    int64_t C = 32;
    int64_t grid_h = 10;
    int64_t grid_w = 10;
    int64_t dim = 2;
    std::initializer_list<int64_t> xShape = {N, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "border",
        true,
        true,
        1,
        status,
        2201220,
        "48 2 32 0 10 10 0 10 10 0 1 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_1)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "zeros",
        true,
        1,
        1,
        status,
        1010320,
        "48 2 1 200 200 200 2 2 2 0 0 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_2)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT16,
        "bilinear",
        "zeros",
        true,
        1,
        1,
        status,
        1010310,
        "48 2 1 200 200 200 2 2 2 0 0 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_nearest)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "nearest",
        "zeros",
        true,
        1,
        1,
        status,
        1010321,
        "48 2 1 200 200 200 2 2 2 1 0 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_nearest)
{
    int64_t N = 2;
    int64_t x_d = 200;
    int64_t x_h = 200;
    int64_t x_w = 200;
    int64_t C = 1;
    int64_t grid_d = 2;
    int64_t grid_h = 2;
    int64_t grid_w = 2;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, x_d, x_h, x_w, C};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, grid_d, grid_h, grid_w, C};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT16,
        "nearest",
        "zeros",
        true,
        1,
        1,
        status,
        1010311,
        "48 2 1 200 200 200 2 2 2 1 0 1 1 2 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float16_bilinear_te)
{
    int64_t N = 88;
    int64_t x_d = 16;
    int64_t x_h = 64;
    int64_t x_w = 64;
    int64_t C = 4;
    int64_t grid_d = 16;
    int64_t grid_h = 64;
    int64_t grid_w = 64;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, C, x_d, x_h, x_w};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, C, grid_d, grid_h, grid_w};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT16,
        "bilinear",
        "zeros",
        true,
        false,
        1,
        status,
        1011310,
        "48 88 4 16 64 64 16 64 64 0 0 1 0 48 0 0 0 ");
}

TEST_F(GridSampleTiling, grid_sample_3d_tiling_test_float32_bilinear_te)
{
    int64_t N = 22;
    int64_t x_d = 16;
    int64_t x_h = 64;
    int64_t x_w = 64;
    int64_t C = 4;
    int64_t grid_d = 16;
    int64_t grid_h = 64;
    int64_t grid_w = 64;
    int64_t dim = 3;
    std::initializer_list<int64_t> xShape = {N, C, x_d, x_h, x_w};
    std::initializer_list<int64_t> gridShape = {N, grid_d, grid_h, grid_w, dim};
    std::initializer_list<int64_t> outShape = {N, C, grid_d, grid_h, grid_w};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(xShape,
        gridShape,
        outShape,
        ge::DT_FLOAT,
        "bilinear",
        "zeros",
        true,
        false,
        1,
        status,
        1011320,
        "48 22 4 16 64 64 16 64 64 0 0 1 0 48 0 0 0 ");
}