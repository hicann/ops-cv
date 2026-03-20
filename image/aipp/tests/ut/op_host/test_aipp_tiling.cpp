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
#include <stdint.h>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <filesystem>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/arch35/aipp_tiling.h"
#include "../../../op_kernel/arch35/aipp_struct.h"

#include "tiling_context_faker.h"

using namespace std;

#define DO_TILING(tilingContextPara)                                                                                   \
    auto contextFaker = gert::TilingContextFaker();                                                                    \
    /* 1. input/output information */                                                                                  \
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();                                                       \
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();                                                     \
    if (tilingContextPara.inputInstanceNum_.size() != 0 || tilingContextPara.outputInstanceNum_.size() != 0) {         \
        contextFaker.IrInstanceNum(tilingContextPara.inputInstanceNum_, tilingContextPara.outputInstanceNum_);         \
    } else {                                                                                                           \
        contextFaker.NodeIoNum(inputNum, outputNum);                                                                   \
    }                                                                                                                  \
    std::vector<gert::Tensor *> inputTensors = {};                                                                     \
    std::vector<gert::Tensor *> outputTensors = {};                                                                    \
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive = {};                                             \
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorsKeepAlive = {};                                            \
    for (size_t index = 0; index < inputNum; index++) {                                                                \
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(                                      \
            tilingContextPara.inputTensorDesc_[index].shape_,                                                          \
            gert::StorageFormat(tilingContextPara.inputTensorDesc_[index].format_,                                     \
             tilingContextPara.inputTensorDesc_[index].format_,                                                        \
             gert::ExpandDimsType()),                                                                                  \
            gert::TensorPlacement::kOnHost,                                                                            \
            tilingContextPara.inputTensorDesc_[index].dtype_,                                                          \
            tilingContextPara.inputTensorDesc_[index].isConst_ ?                                                       \
            tilingContextPara.inputTensorDesc_[index].constValue_:                                                     \
            nullptr);                                                                                                  \
        inputTensors.push_back(curTensor.get());                                                                       \
        inputTensorsKeepAlive.push_back(std::move(curTensor));                                                         \
    }                                                                                                                  \
    for (size_t index = 0; index < outputNum; index++) {                                                               \
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(                                      \
            tilingContextPara.outputTensorDesc_[index].shape_,                                                         \
            gert::StorageFormat(tilingContextPara.outputTensorDesc_[index].format_,                                    \
             tilingContextPara.outputTensorDesc_[index].format_,                                                       \
             gert::ExpandDimsType()),                                                                                  \
            gert::TensorPlacement::kOnHost,                                                                            \
            tilingContextPara.outputTensorDesc_[index].dtype_,                                                         \
            tilingContextPara.outputTensorDesc_[index].isConst_ ?                                                      \
            tilingContextPara.outputTensorDesc_[index].constValue_:                                                    \
            nullptr);                                                                                                  \
        outputTensors.push_back(curTensor.get());                                                                      \
        outputTensorsKeepAlive.push_back(std::move(curTensor));                                                        \
    }                                                                                                                  \
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);                                              \
    for (auto& attrInfo : tilingContextPara.attrs_) {                                                                  \
        switch (attrInfo.attr_.type_) {                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_BOOL: {                                                              \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));       \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_INT: {                                                               \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));    \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_FLOAT: {                                                             \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));      \
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_STRING: {                                                            \
                contextFaker.Attr(attrInfo.attrName_, ge::AscendString(reinterpret_cast<std::string*>(attrInfo.attr_.valuePtr_.get())->c_str()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_BOOL: {                                                         \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<bool>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_INT: {                                                          \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<int64_t>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_LIST_INT: {                                                     \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Cv::AnyValue::ValueType::VT_LIST_FLOAT: {                                                        \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<float>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            default:                                                                                                   \
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_ << "is not supported!" << std::endl;\
        }                                                                                                              \
    }                                                                                                                  \
    /* 2. base information */                                                                                          \
    fe::PlatFormInfos platformInfo;                                                                                    \
    platformInfo.Init();                                                                                               \
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);                                  \
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);                                                     \
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str())                                     \
                                     .CompileInfo(tilingContextPara.compileInfo_)                                      \
                                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))                             \
                                     .TilingData(tilingData.get())                                                     \
                                     .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()))           \
                                     .Build();                                                                         \
    string compileInfoStringPrefix = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )";\
    string compileInfoStringMiddle = R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )";\
    string compileInfoStringSuffix = R"(} })";\
    string compileInfoString = compileInfoStringPrefix +                                                               \
                               std::to_string(tilingContextPara.ubSize_) +                                             \
                               compileInfoStringMiddle +                                                               \
                               std::to_string(tilingContextPara.coreNum_) +                                            \
                               compileInfoStringSuffix;                                                                \
     map<string, string> socToArch = {                                                                                 \
        {"Ascend310P", "2002"},                                                                                        \
        {"Ascend910B", "2201"},                                                                                        \
        {"Ascend910_93", "2201"},                                                                                      \
        {"Ascend950", "3510"},                                                                                         \
        {"Ascend910", "1001"}                                                                                          \
    };                                                                                                                 \
    map<string, string> socInfos;                                                                                      \
    map<string, string> aicoreSpec;                                                                                    \
    map<string, string> intrinsics;                                                                                    \
    map<string, string> socversions = {                                                                                \
        {"NpuArch", socToArch[tilingContextPara.socVersion_]},                                                         \
        {"Short_SoC_version", tilingContextPara.socVersion_}                                                           \
    };                                                                                                                 \
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);                                     \
    auto tilingContext = contextHolder.GetContext();                                                                   \
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);                                             \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);                                        \
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                  \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);                           \
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);                                          \
    /* 3. get tiling func */                                                                                           \
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();                         \
    auto tilingFunc = spaceRegistry->GetOpImpl(tilingContextPara.opName_.c_str())->tiling;                             \
    /* 4. check tiling func */                                                                                         \
    auto tilingRet = tilingFunc(tilingContext);

template <typename T>
static string to_string(void* buf, size_t size) {
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

class AippTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AippTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AippTiling TearDown" << std::endl;
    }
};

struct AippCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
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

void AippExecuteTestCase(const gert::TilingContextPara& tilingContextPara, 
                     ge::graphStatus                expectResult,
                     uint64_t                       expectTilingKey, 
                     const string&                  expectTilingData,
                     const std::vector<size_t>&     expectWorkspaces)
{
    DO_TILING(tilingContextPara)

    // check tiling func
    EXPECT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
        }
    }

    // check tiling key
    auto tilingKeyResult = tilingContext->GetTilingKey();
    // fixme
    ASSERT_EQ(tilingKeyResult, expectTilingKey);

    // check tiling data
    auto buf = (AippTilingData*)tilingContext->GetRawTilingData()->GetData();
    AippTilingData tilingParam = *buf;
    std::stringstream ss;
    ss << (int)tilingParam.imageFormat << " " << (int)tilingParam.outputFormat
       << " " << tilingParam.batchNum << " " << tilingParam.channelNum
       << " " << tilingParam.inputSizeH << " " << tilingParam.inputSizeW
       << " " << tilingParam.cropParam.cropSwitch << " " << tilingParam.cropParam.cropStartPosH 
       << " " << tilingParam.cropParam.cropStartPosW << " " << tilingParam.cropParam.cropSizeH 
       << " " << tilingParam.cropParam.cropSizeW << " " << tilingParam.cscParam.cscSwitch
       << " " << tilingParam.cscParam.cscMatrix00 << " " << tilingParam.cscParam.cscMatrix01
       << " " << tilingParam.cscParam.cscMatrix02 << " " << tilingParam.cscParam.cscMatrix10
       << " " << tilingParam.cscParam.cscMatrix11 << " " << tilingParam.cscParam.cscMatrix12
       << " " << tilingParam.cscParam.cscMatrix20 << " " << tilingParam.cscParam.cscMatrix21
       << " " << tilingParam.cscParam.cscMatrix22 << " " << tilingParam.cscParam.outBias0
       << " " << tilingParam.cscParam.outBias1 << " " << tilingParam.cscParam.outBias2
       << " " << tilingParam.cscParam.inBias0 << " " << tilingParam.cscParam.inBias1
       << " " << tilingParam.cscParam.inBias2
       << " " << tilingParam.dtcParam.dtcPixelMeanChn0
       << " " << tilingParam.dtcParam.dtcPixelMeanChn1
       << " " << tilingParam.dtcParam.dtcPixelMeanChn2
       << " " << tilingParam.dtcParam.dtcPixelMeanChn3
       << " " << tilingParam.dtcParam.dtcPixelMinChn0 
       << " " << tilingParam.dtcParam.dtcPixelMinChn1
       << " " << tilingParam.dtcParam.dtcPixelMinChn2 
       << " " << tilingParam.dtcParam.dtcPixelMinChn3
       << " " << tilingParam.dtcParam.dtcPixelVarReciChn0
       << " " << tilingParam.dtcParam.dtcPixelVarReciChn1
       << " " << tilingParam.dtcParam.dtcPixelVarReciChn2
       << " " << tilingParam.dtcParam.dtcPixelVarReciChn3;

    auto tilingDataResult = ss.str();
    // fixme
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(AippTiling, aipp_tiling_test_0)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{10, 256, 224, 3}, {10, 256, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{10, 3, 256, 224}, {10, 3, 256, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo);

    uint64_t expectTilingKey = 1;
    string expectTilingData = "2 1 10 3 256 224 0 0 0 256 224 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_1)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{100, 2560, 2240, 3}, {100, 2560, 2240, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{100, 256, 224, 3}, {100, 256, 224, 3}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", 
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","crop":true,"crop_size_h":256,"crop_size_w":224,
        "input_format":"RGB888_U8","load_start_pos_h":16,"load_start_pos_w":16,
        "min_chn_0":123.67500305175781,"min_chn_1":116.27999877929688,
        "min_chn_2":103.52999877929688,"src_image_size_h":2560,"src_image_size_w":2240,
        "var_reci_chn_0":0.017124753445386887,"var_reci_chn_1":0.017507003620266914,
        "var_reci_chn_2":0.01742919348180294, "csc_switch":true, "matrix_r0c0":256,
        "matrix_r0c1":101, "matrix_r0c2":-202, "matrix_r1c0":301, "matrix_r1c1":-256,
        "matrix_r1c2":402, "matrix_r2c0":503, "matrix_r2c1":601, "matrix_r2c2":256,
        "output_bias_0":110, "output_bias_1":120, "output_bias_2":83})"))},
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = \
        "2 2 100 3 2560 2240 1 16 16 256 224 1 256 101 -202 301 -256 402 503 601 256 110 120 83 0 0 0 0 0 0 0 123.675 116.28 103.53 0 0.0171248 0.017507 0.0174292 1";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_2)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_01.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_01.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 224, 224}, {1, 3, 224, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    uint64_t expectTilingKey = 1;
    string expectTilingData = "2 1 1 3 224 224 0 0 0 224 224 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 128 128 0 0 0 0 0 0.00781 0.00781 0.00781 1";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_3)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_02.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_02.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 1080, 3}, {1, 256, 1080, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 224, 256}, {1, 3, 224, 256}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "2 1 1 3 1080 256 1 2 16 256 224 1 298 10 409 298 -100 -208 298 516 20 110 120 83 0 0 0 10 20 30 0 123.675 116.28 103.53 0 0.0171248 0.017507 0.0174292 1";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_4)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_03.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_03.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 256, 3}, {1, 256, 256, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 256, 256, 3}, {1, 256, 256, 3}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    uint64_t expectTilingKey = 2;
    string expectTilingData = "1 2 1 3 256 256 0 0 0 256 256 1 298 0 409 298 -100 -208 298 516 0 110 120 83 110 120 83 10 20 30 0 123.675 116.28 103.53 0 0.0171248 0.017507 0.0174292 1";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_5)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{10, 3, 256, 224}, {10, 3, 256, 224}}, ge::DT_UINT8, ge::FORMAT_NCHW}},
        {{{{10, 3, 256, 224}, {10, 3, 256, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_6)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 221, 3}, {1, 256, 221, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 256, 224}, {1, 3, 256, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"YUV420SP_U8"})"))},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_7)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 222, 3}, {1, 256, 222, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 256, 224}, {1, 3, 256, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"YUV420SP_U8",
        "crop":true,"load_start_pos_h":222,"load_start_pos_w":21})"))},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_8)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 4097, 222, 3}, {1, 4097, 222, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 256, 224}, {1, 3, 256, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo);

    uint64_t expectTilingKey = 1;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
