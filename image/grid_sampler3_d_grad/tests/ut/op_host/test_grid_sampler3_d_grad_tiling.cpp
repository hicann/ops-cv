/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../op_host/grid_sampler3_d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampler3DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GridSampler3DGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GridSampler3DGradTiling TearDown" << std::endl;
    }
};

struct GridSampler3DGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isDavid{false};
};

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("GridSampler3DGrad",
                                                {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {x, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{dx, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
                                                gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 0 206158430209 68719476757 1099511627777 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case2)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case3)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("border")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case4)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
//     gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("reflection")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case5)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
//     gert::StorageShape x = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
//     gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     gert::StorageShape dx = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
//     gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case6)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{1, 300, 2, 2, 2}, {1, 300, 2, 2, 2}};
//     gert::StorageShape x = {{1, 300, 2, 2, 2}, {1, 300, 2, 2, 2}};
//     gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     gert::StorageShape dx = {{1, 300, 2, 2, 2}, {1, 300, 2, 2, 2}};
//     gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case7)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{1, 600, 2, 2, 2}, {1, 600, 2, 2, 2}};
//     gert::StorageShape x = {{1, 600, 2, 2, 2}, {1, 600, 2, 2, 2}};
//     gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     gert::StorageShape dx = {{1, 600, 2, 2, 2}, {1, 600, 2, 2, 2}};
//     gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case8)
// {
//     std::string op_type("GridSampler3DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler3DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{1, 1600, 2, 2, 2}, {1, 1600, 2, 2, 2}};
//     gert::StorageShape x = {{1, 1600, 2, 2, 2}, {1, 1600, 2, 2, 2}};
//     gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     gert::StorageShape dx = {{1, 1600, 2, 2, 2}, {1, 1600, 2, 2, 2}};
//     gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&dx, &dgrid})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }