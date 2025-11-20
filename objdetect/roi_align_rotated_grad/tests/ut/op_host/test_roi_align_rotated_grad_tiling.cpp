/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_align_rotated_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "../../../op_host/roi_align_rotated_grad_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

class TilingForRoiAlignRotatedGrad : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "TilingForRoiAlignRotatedGrad SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "TilingForRoiAlignRotatedGrad TearDown" << std::endl;
  }
};

static void GetPlatFormInfos(const char* compileInfoStr, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                             map<string, string>& intrinsics) {
    string default_hardward_info = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,
                          "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "CORE_NUM": 32}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    if (compileInfoJson.type() != nlohmann::json::value_t::object) {
        compileInfoJson = nlohmann::json::parse(default_hardward_info.c_str());
    }

    map<string, string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"},
                                       {"l2_size", "L2_SIZE"},
                                       {"cube_core_cnt", "cube_core_cnt"},
                                       {"vector_core_cnt", "vector_core_cnt"},
                                       {"core_type_list", "core_type_list"}};
    socInfos["core_type_list"] = "AICore";

    for (auto &t : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            auto &objJson = compileInfoJson["hardware_info"][t.second];
            if (objJson.is_number_integer()) {
                socInfos[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            } else if (objJson.is_string()) {
                socInfos[t.first] = objJson;
            }
        }
    }
    map<string, string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"},
                                          {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"},
                                          {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},
                                          {"bt_size", "BT_SIZE"},
                                          {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto &t : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            if (t.second == "load3d_constraints") {
                aicoreSpec[t.first] = compileInfoJson["hardware_info"][t.second].get<string>();
            } else {
                aicoreSpec[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            }
        }
    }

    std::string intrinsicsKeys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub",
                                    "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                    "Intrinsic_matmul_ub_to_ub", "Intrinsic_conv_ub_to_ub",
                                    "Intrinsic_data_move_l12bt"};
    for (string key : intrinsicsKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = "float16";
            if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
                intrinsics[key] = "bf16";
            }
        }
    }
}

#define DO_TILING(tilingPara)                                                                                          \
    auto contextFaker = gert::TilingContextFaker();                                                                    \
    /* 1. input/output information */                                                                                  \
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();                                                       \
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();                                                     \
    contextFaker.NodeIoNum(inputNum, outputNum);                                                                       \
    std::vector<gert::Tensor *> inputTensors = {};                                                                     \
    std::vector<gert::Tensor *> outputTensors = {};                                                                    \
    for (size_t index = 0; index < inputNum; index++) {                                                                \
        contextFaker.NodeInputTd(index,                                                                                \
                                 tilingContextPara.inputTensorDesc_[index].dtype_,                                     \
                                 tilingContextPara.inputTensorDesc_[index].format_,                                    \
                                 tilingContextPara.inputTensorDesc_[index].format_);                                   \
        inputTensors.push_back((gert::Tensor *)&tilingContextPara.inputTensorDesc_[index].shape_);                     \
    }                                                                                                                  \
    for (size_t index = 0; index < outputNum; index++) {                                                               \
        contextFaker.NodeOutputTd(index,                                                                               \
                                  tilingContextPara.outputTensorDesc_[index].dtype_,                                   \
                                  tilingContextPara.outputTensorDesc_[index].format_,                                  \
                                  tilingContextPara.outputTensorDesc_[index].format_);                                 \
        outputTensors.push_back((gert::Tensor *)&tilingContextPara.outputTensorDesc_[index].shape_);                   \
    }                                                                                                                  \
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);                                              \
    for (auto& attrInfo : tilingContextPara.attrs_) {                                                                  \
        switch (attrInfo.attr_.type_) {                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_BOOL: {                                                            \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));       \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_INT: {                                                             \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));    \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_FLOAT: {                                                           \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));      \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_STRING: {                                                          \
                contextFaker.Attr(attrInfo.attrName_, AscendString(reinterpret_cast<std::string*>(attrInfo.attr_.valuePtr_.get())->c_str()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_BOOL: {                                                       \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<bool>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_INT: {                                                        \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<int64_t>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_LIST_INT: {                                                   \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_FLOAT: {                                                      \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<float>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            default:                                                                                                   \
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_ << "is not supported!" << std::endl;\
        }                                                                                                              \
    }                                                                                                                   \
    /* 2. base information */                                                                                          \
    fe::PlatFormInfos platformInfo;                                                                                    \
    platformInfo.Init();                                                                                               \
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);                                  \
    gert::ContinuousVector workspace;                                                                                  \
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str())                                     \
                                     .CompileInfo(tilingContextPara.compileInfo_)                                      \
                                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))                             \
                                     .TilingData(tilingData.get())                                                     \
                                     .Workspace(&workspace)                                                            \
                                     .Build();                                                                         \
    string compileInfoStringPrefix = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )";\
    string compileInfoStringMiddle = R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )";\
    string compileInfoStringSuffix = R"(, "socVersion": "Ascend910_95"} })";\
    string compileInfoString = compileInfoStringPrefix +                                                               \
                               std::to_string(tilingContextPara.ubSize_) +                                             \
                               compileInfoStringMiddle +                                                               \
                               std::to_string(tilingContextPara.coreNum_) +                                            \
                               compileInfoStringSuffix;                                                                \
    map<string, string> socInfos;                                                                                      \
    map<string, string> aicoreSpec;                                                                                    \
    map<string, string> intrinsics;                                                                                    \
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};                                         \
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);                                     \
    auto tilingContext = contextHolder.GetContext();                                                                   \
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);                                             \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);                                        \
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                  \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);                           \
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);                                          \


TEST_F(TilingForRoiAlignRotatedGrad, roi_align_rotated_grad_tiling_0)
{
    optiling::RoiAlignRotatedGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("RoiAlignRotatedGrad",
                                                {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
                                                {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{8, 2, 2, 8}, {8, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("y_grad_shape", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({8, 2, 2, 8})),
                                                gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
                                                gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
                                                gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
                                                gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("clockwise", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
                                                &compileInfo);

    DO_TILING(tilingContextPara)

    auto tilingKeyResult = tilingContext->GetTilingKey();

    uint64_t expectTilingKey = tilingKeyResult;
    string expectTilingData = "34359738368 8589934600 34359738370 8589934600 2 4539628424389459969 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(TilingForRoiAlignRotatedGrad, roi_align_rotated_grad_tiling_0)
// {
//   std::string op_type("RoiAlignRotatedGrad");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
//   string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 48}
//                                     })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   static optiling::RoiAlignRotatedGradCompileInfo compile_info = {40, 196352};
//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();
//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);
//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096 * 16);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096 * 16);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//   gert::StorageShape input_0 = {{8, 8, 8, 8}, {8, 8, 8, 8}};
//   gert::StorageShape input_1 = {{6, 8}, {6, 8}};
//   gert::StorageShape output_shape = {{8, 2, 2, 8}, {8, 2, 2, 8}};

//   std::vector<int64_t> y_grad_shape = {8, 2, 2, 8};

//   // tilingParseFunc simulate
//   auto holder =
//       gert::TilingContextFaker()
//           .NodeIoNum(2, 1)
//           .IrInstanceNum({1, 1})
//           .InputShapes({&input_0, &input_1})
//           .OutputShapes({&output_shape})
//           .CompileInfo(&compile_info)
//           .NodeAttrs({{"y_grad_shape", ge::AnyValue::CreateFrom<vector<int64_t>>(y_grad_shape)},
//                       {"pooled_h", ge::AnyValue::CreateFrom<int64_t>(2)},
//                       {"pooled_w", ge::AnyValue::CreateFrom<int64_t>(2)},
//                       {"spatial_scale", ge::AnyValue::CreateFrom<float>(0.5)},
//                       {"sampling_ratio", ge::AnyValue::CreateFrom<int64_t>(1)},
//                       {"aligned", ge::AnyValue::CreateFrom<bool>(false)},
//                       {"clockwise", ge::AnyValue::CreateFrom<bool>(false)}})
//           .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(ws_size)
//           .Build();

//   gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   ASSERT_EQ(tiling_key, 0);
// }