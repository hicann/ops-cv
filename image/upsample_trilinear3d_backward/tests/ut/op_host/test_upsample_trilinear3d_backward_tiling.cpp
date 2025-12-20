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

#include "../../../op_host/upsample_trilinear3d_backward_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace gert;
using namespace optiling;

class UpsampleTrilinear3dBackwardTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleTrilinear3dBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleTrilinear3dBackwardTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleTrilinear3dBackwardTiling, upsample_trilinear3d_tiling_001)
{
    UpsampleTrilinearBackwardCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("UpsampleTrilinear3dBackward",
        {{{{1, 1, 4, 128, 128}, {1, 1, 4, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 12, 256, 256}, {1, 1, 12, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({12, 256, 256})),
        gert::TilingContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 4, 128, 128})),
        gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
        gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2 1 12 256 256 4 128 128 4611686019501129728 72340169604202496 16 224 131072 262144 0 0 0 1 16 16 0 0 0 1 1 3 1 4 512 16 16 171 1 16 48 2199023255553 549755814144 2199023255680 60129542160 68719477248 4294967312 4294967297 1 0 145135534866432 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 68719476864 60129542400 1099511627792 4294967312 4294967297 1 0 74766790688768 16384 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 51539607553 17179934720 68719476740 17179934720 1099511627792 4294967304 4294967312 8 4294967296 565148976676864 16384 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {35170304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(UpsampleTrilinear3dBackwardTiling, upsample_trilinear3d_tiling_002)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}};
//     gert::StorageShape out_shape = {{1, 1, 0, 0, 0}, {1, 1, 0, 0, 0}};

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
//     struct UpsampleTrilinearBackwardCompileInfo {
//         int64_t coreNum = 20;
//     } compile_info;

//     std::string op_type("UpsampleTrilinear3dBackward");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {0, 0, 0};
//     std::vector<int64_t> input_size = {1, 1, 128, 128, 128};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleTrilinear3dBackward")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeAttrs(
//                           {{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                            {"input_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                            {"scales_d", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(UpsampleTrilinear3dBackwardTiling, upsample_trilinear3d_tiling_003)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 2, 60, 256, 512}, {1, 2, 60, 256, 512}};
//     gert::StorageShape out_shape = {{1, 2, 4, 1, 2}, {1, 2, 4, 1, 2}};

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
//     struct UpsampleTrilinearBackwardCompileInfo {
//         int64_t coreNum = 20;
//     } compile_info;

//     std::string op_type("UpsampleTrilinear3dBackward");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {4, 1, 2};
//     std::vector<int64_t> input_size = {1, 2, 60, 256, 512};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("UpsampleTrilinear3dBackward")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeAttrs(
//                           {{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                            {"input_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                            {"scales_d", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }
