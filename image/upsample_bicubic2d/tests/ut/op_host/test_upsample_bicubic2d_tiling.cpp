/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bicubic2d_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class UpsampleBicubic2dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBicubic2dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBicubic2dTiling TearDown" << std::endl;
    }
};

struct UpsampleBicubic2dCompileInfo {
    uint32_t coreNum = 0;
    uint16_t socVersionType = 0;
};

TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_001)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};
    UpsampleBicubic2dCompileInfo compileInfo = {48, 220};
    std::vector<int64_t> output_size = {256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "64 2 4539628425446424576 17180131328 131072 9895604652288 17179869192 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 274877906944 274877906944 274877906944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 549755813952 549755813952 549755813952 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 824633720960 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 1099511627968 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 154618822720 274877907072 4294967336 4294967297 1 0 131941395333120 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 154618822912 1099511627840 8589934624 4294967298 1 0 351843720888320 65536 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33759232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_002)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
//     gert::StorageShape out_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};

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
//     struct UpsampleBicubic2dCompileInfo {
//         uint16_t totalCoreNum = 0;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("UpsampleBicubic2d");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
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
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {128, 128};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleBicubic2d")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeAttrs({{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                           {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                           {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                           {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     // todo check tiling result
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_003)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 128, 128, 111}, {1, 1, 128, 128, 111}};
//     gert::StorageShape out_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};

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
//     struct UpsampleBicubic2dCompileInfo {
//         uint16_t totalCoreNum = 0;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("UpsampleBicubic2d");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
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
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {0, 0};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleBicubic2d")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeAttrs({{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                           {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                           {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                           {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     ASSERT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
// }

// // 310P test
// TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_004)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
//     gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};

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
//     map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     struct UpsampleBicubic2dCompileInfo {
//         uint16_t totalCoreNum = 0;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("UpsampleBicubic2d");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {256, 256};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleBicubic2d")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeAttrs({{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                           {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                           {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                           {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     // todo check tiling result
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_005)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
//     gert::StorageShape out_shape = {{1, 1, 16, 16}, {1, 1, 16, 16}};

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
//     struct UpsampleBicubic2dCompileInfo {
//         uint16_t totalCoreNum = 0;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("UpsampleBicubic2d");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
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
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {16, 16};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleBicubic2d")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeAttrs({{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                           {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                           {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                           {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     // todo check tiling result
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }