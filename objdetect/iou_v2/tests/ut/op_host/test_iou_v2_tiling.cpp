/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_iou_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/iou_v2_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class IouV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "IouV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IouV2Tiling TearDown" << std::endl;
    }
};


struct IouV2CompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(IouV2Tiling, test_tiling_iou_aligned_f32)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("IouV2",
                                                {{{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1024, 1}, {1024, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
                                                gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
                                                gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
                                                &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "1024 1024 0 1 16 16 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


// TEST_F(IouV2Tiling, test_tiling_iou_not_aligned_f16)
// {
//     gert::StorageShape bboxes_shape = {{1024, 4}, {1024, 4}};
//     gert::StorageShape gtboxes_shape = {{200, 4}, {200, 4}};

//     // output
//     gert::StorageShape overlap_shape = {{200, 1024}, {200, 1024}};
//     string compile_info_string = R"({
//           "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                             "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
//                             "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                             "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                             "CORE_NUM": 40}
//                             })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();

//     // compile info
//     struct IouV2CompileInfo
//     {
//     };
//     IouV2CompileInfo compile_info;

//     std::string op_type("IouV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder = gert::KernelRunContextFaker()
//                              .KernelIONum(2, 1)
//                              .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
//                              .Outputs({&compile_info})
//                              .Build();

//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("IouV2")
//                       .NodeIoNum(2, 1)
//                       .IrInstanceNum({1, 1})
//                       .InputShapes({&bboxes_shape, &gtboxes_shape})
//                       .OutputShapes({&overlap_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<std::string>("iou")},
//                                   {"eps", ge::AnyValue::CreateFrom<float>(0.0)},
//                                   {"aligned", ge::AnyValue::CreateFrom<bool>(false)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context, nullptr);
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 8);
// }

// TEST_F(IouV2Tiling, test_tiling_iof_aligned_f32)
// {
//     gert::StorageShape bboxes_shape = {{4, 800}, {4, 800}};
//     gert::StorageShape gtboxes_shape = {{4, 800}, {4, 800}};

//     // output
//     gert::StorageShape overlap_shape = {{800, 1}, {800, 1}};
//     string compile_info_string = R"({
//           "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                             "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
//                             "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                             "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                             "CORE_NUM": 40}
//                             })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();

//     // compile info
//     struct IouV2CompileInfo
//     {
//     };
//     IouV2CompileInfo compile_info;

//     std::string op_type("IouV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder = gert::KernelRunContextFaker()
//                              .KernelIONum(2, 1)
//                              .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
//                              .Outputs({&compile_info})
//                              .Build();

//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("IouV2")
//                       .NodeIoNum(2, 1)
//                       .IrInstanceNum({1, 1})
//                       .InputShapes({&bboxes_shape, &gtboxes_shape})
//                       .OutputShapes({&overlap_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<std::string>("iof")},
//                                   {"eps", ge::AnyValue::CreateFrom<float>(0.0)},
//                                   {"aligned", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context, nullptr);
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 14);
// }

// TEST_F(IouV2Tiling, test_tiling_iof_not_aligned_bf16)
// {
//     gert::StorageShape bboxes_shape = {{1024, 4}, {1024, 4}};
//     gert::StorageShape gtboxes_shape = {{200, 4}, {200, 4}};

//     // output
//     gert::StorageShape overlap_shape = {{1024, 200}, {1024, 200}};
//     string compile_info_string = R"({
//           "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                             "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
//                             "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                             "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                             "CORE_NUM": 40}
//                             })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();

//     // compile info
//     struct IouV2CompileInfo
//     {
//     };
//     IouV2CompileInfo compile_info;

//     std::string op_type("IouV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder = gert::KernelRunContextFaker()
//                              .KernelIONum(2, 1)
//                              .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
//                              .Outputs({&compile_info})
//                              .Build();

//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("IouV2")
//                       .NodeIoNum(2, 1)
//                       .IrInstanceNum({1, 1})
//                       .InputShapes({&bboxes_shape, &gtboxes_shape})
//                       .OutputShapes({&overlap_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                       .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<std::string>("iof")},
//                                   {"eps", ge::AnyValue::CreateFrom<float>(0.0)},
//                                   {"aligned", ge::AnyValue::CreateFrom<bool>(false)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context, nullptr);
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 19);
// }