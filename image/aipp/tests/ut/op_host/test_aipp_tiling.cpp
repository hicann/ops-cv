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

void CompareAippTilingData(const Aipp_Kernel::AippTilingData& actual,
                           const Aipp_Kernel::AippTilingData& expected) 
{
    EXPECT_EQ(actual.imageFormat, expected.imageFormat);
    EXPECT_EQ(actual.outputFormat, expected.outputFormat);
    EXPECT_EQ(actual.batchNum, expected.batchNum);
    EXPECT_EQ(actual.channelNum, expected.channelNum);
    EXPECT_EQ(actual.inputSizeH, expected.inputSizeH);
    EXPECT_EQ(actual.inputSizeW, expected.inputSizeW);
    EXPECT_EQ(actual.outputSizeH, expected.outputSizeH);
    EXPECT_EQ(actual.outputSizeW, expected.outputSizeW);
    EXPECT_EQ(actual.srcChannelOffset, expected.srcChannelOffset);

    // CscParam
    EXPECT_EQ(actual.cscParam.cscSwitch, expected.cscParam.cscSwitch);
    EXPECT_EQ(actual.cscParam.rbuvSwapSwitch, expected.cscParam.rbuvSwapSwitch);
    EXPECT_EQ(actual.cscParam.axSwapSwitch, expected.cscParam.axSwapSwitch);
    EXPECT_EQ(actual.cscParam.cscMatrix00, expected.cscParam.cscMatrix00);
    EXPECT_EQ(actual.cscParam.cscMatrix01, expected.cscParam.cscMatrix01);
    EXPECT_EQ(actual.cscParam.cscMatrix02, expected.cscParam.cscMatrix02);
    EXPECT_EQ(actual.cscParam.cscMatrix10, expected.cscParam.cscMatrix10);
    EXPECT_EQ(actual.cscParam.cscMatrix11, expected.cscParam.cscMatrix11);
    EXPECT_EQ(actual.cscParam.cscMatrix12, expected.cscParam.cscMatrix12);
    EXPECT_EQ(actual.cscParam.cscMatrix20, expected.cscParam.cscMatrix20);
    EXPECT_EQ(actual.cscParam.cscMatrix21, expected.cscParam.cscMatrix21);
    EXPECT_EQ(actual.cscParam.cscMatrix22, expected.cscParam.cscMatrix22);
    EXPECT_EQ(actual.cscParam.outBias0, expected.cscParam.outBias0);
    EXPECT_EQ(actual.cscParam.outBias1, expected.cscParam.outBias1);
    EXPECT_EQ(actual.cscParam.outBias2, expected.cscParam.outBias2);
    EXPECT_EQ(actual.cscParam.inBias0, expected.cscParam.inBias0);
    EXPECT_EQ(actual.cscParam.inBias1, expected.cscParam.inBias1);
    EXPECT_EQ(actual.cscParam.inBias2, expected.cscParam.inBias2);

    // CropParam
    EXPECT_EQ(actual.cropParam.cropSwitch, expected.cropParam.cropSwitch);
    EXPECT_EQ(actual.cropParam.cropStartPosH, expected.cropParam.cropStartPosH);
    EXPECT_EQ(actual.cropParam.cropStartPosW, expected.cropParam.cropStartPosW);
    EXPECT_EQ(actual.cropParam.cropSizeH, expected.cropParam.cropSizeH);
    EXPECT_EQ(actual.cropParam.cropSizeW, expected.cropParam.cropSizeW);

    // DtcParam
    EXPECT_EQ(actual.dtcParam.dtcPixelMeanChn0, expected.dtcParam.dtcPixelMeanChn0);
    EXPECT_EQ(actual.dtcParam.dtcPixelMeanChn1, expected.dtcParam.dtcPixelMeanChn1);
    EXPECT_EQ(actual.dtcParam.dtcPixelMeanChn2, expected.dtcParam.dtcPixelMeanChn2);
    EXPECT_EQ(actual.dtcParam.dtcPixelMeanChn3, expected.dtcParam.dtcPixelMeanChn3);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelMinChn0, expected.dtcParam.dtcPixelMinChn0);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelMinChn1, expected.dtcParam.dtcPixelMinChn1);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelMinChn2, expected.dtcParam.dtcPixelMinChn2);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelMinChn3, expected.dtcParam.dtcPixelMinChn3);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelVarReciChn0, expected.dtcParam.dtcPixelVarReciChn0);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelVarReciChn1, expected.dtcParam.dtcPixelVarReciChn1);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelVarReciChn2, expected.dtcParam.dtcPixelVarReciChn2);
    EXPECT_FLOAT_EQ(actual.dtcParam.dtcPixelVarReciChn3, expected.dtcParam.dtcPixelVarReciChn3);

    // PaddingParam
    EXPECT_EQ(actual.paddingParam.paddingSwitch, expected.paddingParam.paddingSwitch);
    EXPECT_EQ(actual.paddingParam.leftPaddingSize, expected.paddingParam.leftPaddingSize);
    EXPECT_EQ(actual.paddingParam.rightPaddingSize, expected.paddingParam.rightPaddingSize);
    EXPECT_EQ(actual.paddingParam.topPaddingSize, expected.paddingParam.topPaddingSize);
    EXPECT_EQ(actual.paddingParam.bottomPaddingSize, expected.paddingParam.bottomPaddingSize);
    EXPECT_FLOAT_EQ(actual.paddingParam.padValue, expected.paddingParam.padValue);
}

void AippExecuteTestCase(const gert::TilingContextPara& tilingContextPara,
                         ge::graphStatus expectResult,
                         uint64_t expectTilingKey,
                         const Aipp_Kernel::AippTilingData& expectTiling,
                         const std::vector<size_t>& expectWorkspaces)
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
    ASSERT_EQ(tilingKeyResult, expectTilingKey);

    // check tiling data
    auto buf = reinterpret_cast<const Aipp_Kernel::AippTilingData*>(tilingContext->GetRawTilingData()->GetData());
    Aipp_Kernel::AippTilingData actualTiling = *buf;
    CompareAippTilingData(actualTiling, expectTiling);
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

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 10,
        .channelNum = 3,
        .inputSizeW = 224,
        .inputSizeH = 256,
        .outputSizeW = 224,
        .outputSizeH = 256,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 256,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0,
            .dtcPixelMeanChn1 = 0,
            .dtcPixelMeanChn2 = 0,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1,
            .dtcPixelVarReciChn1 = 1,
            .dtcPixelVarReciChn2 = 1,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 2,
        .batchNum = 100,
        .channelNum = 3,
        .inputSizeW = 2240,
        .inputSizeH = 2560,
        .outputSizeW = 224,
        .outputSizeH = 256,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 101,
            .cscMatrix02 = -202,
            .cscMatrix10 = 301,
            .cscMatrix11 = -256,
            .cscMatrix12 = 402,
            .cscMatrix20 = 503,
            .cscMatrix21 = 601,
            .cscMatrix22 = 256,
            .outBias0 = 110,
            .outBias1 = 120,
            .outBias2 = 83,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 1,
            .cropStartPosH = 16,
            .cropStartPosW = 16,
            .cropSizeH = 256,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0,
            .dtcPixelMeanChn1 = 0,
            .dtcPixelMeanChn2 = 0,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 123.675f,
            .dtcPixelMinChn1 = 116.28f,
            .dtcPixelMinChn2 = 103.53f,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.017124753f,
            .dtcPixelVarReciChn1 = 0.017507f,
            .dtcPixelVarReciChn2 = 0.0174292f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 224,
        .inputSizeH = 224,
        .outputSizeW = 224,
        .outputSizeH = 224,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.00781f,
            .dtcPixelVarReciChn1 = 0.00781f,
            .dtcPixelVarReciChn2 = 0.00781f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 256,
        .inputSizeH = 1080,
        .outputSizeW = 256,
        .outputSizeH = 224,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 298,
            .cscMatrix01 = 10,
            .cscMatrix02 = 409,
            .cscMatrix10 = 298,
            .cscMatrix11 = -100,
            .cscMatrix12 = -208,
            .cscMatrix20 = 298,
            .cscMatrix21 = 516,
            .cscMatrix22 = 20,
            .outBias0 = 110,
            .outBias1 = 120,
            .outBias2 = 83,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 1,
            .cropStartPosH = 2,
            .cropStartPosW = 16,
            .cropSizeH = 256,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 10,
            .dtcPixelMeanChn1 = 20,
            .dtcPixelMeanChn2 = 30,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 123.675f,
            .dtcPixelMinChn1 = 116.28f,
            .dtcPixelMinChn2 = 103.53f,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.017124753f,
            .dtcPixelVarReciChn1 = 0.017507004f,
            .dtcPixelVarReciChn2 = 0.017429193f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_YUV_TO_RGB;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 1,
        .outputFormat = 2,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 256,
        .inputSizeH = 256,
        .outputSizeW = 256,
        .outputSizeH = 256,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 1,
            .axSwapSwitch = 0,
            .cscMatrix00 = 298,
            .cscMatrix01 = 409,
            .cscMatrix02 = 0,
            .cscMatrix10 = 298,
            .cscMatrix11 = -208,
            .cscMatrix12 = -100,
            .cscMatrix20 = 298,
            .cscMatrix21 = 0,
            .cscMatrix22 = 516,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 110,
            .inBias1 = 120,
            .inBias2 = 83
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 256,
            .cropSizeW = 256
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 10,
            .dtcPixelMeanChn1 = 20,
            .dtcPixelMeanChn2 = 30,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 123.675f,
            .dtcPixelMinChn1 = 116.28f,
            .dtcPixelMinChn2 = 103.53f,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.017124753f,
            .dtcPixelVarReciChn1 = 0.017507f,
            .dtcPixelVarReciChn2 = 0.0174292f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_ERROR_TILINGKEY;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_ERROR_TILINGKEY;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_ERROR_TILINGKEY;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
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

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_9)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_04.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_04.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 4}, {1, 224, 224, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 224, 224}, {1, 3, 224, 224}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 4,
        .inputSizeW = 224,
        .inputSizeH = 224,
        .outputSizeW = 224,
        .outputSizeH = 224,
        .srcChannelOffset = 1,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 1,
            .axSwapSwitch = 1,
            .cscMatrix00 = -202,
            .cscMatrix01 = 101,
            .cscMatrix02 = 298,
            .cscMatrix10 = 402,
            .cscMatrix11 = -256,
            .cscMatrix12 = 301,
            .cscMatrix20 = 256,
            .cscMatrix21 = 601,
            .cscMatrix22 = 503,
            .outBias0 = 110,
            .outBias1 = 120,
            .outBias2 = 83,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.00781f,
            .dtcPixelVarReciChn1 = 0.00781f,
            .dtcPixelVarReciChn2 = 0.00781f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0,
            .leftPaddingSize = 0,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_10)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_05.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_05.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 256, 3}, {1, 256, 256, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 256, 256}, {1, 3, 256, 256}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_ERROR_TILINGKEY;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_test_11)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_ERROR_TILINGKEY;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_1)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_06.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_06.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 244, 244}, {1, 3, 244, 244}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 224,
        .inputSizeH = 224,
        .outputSizeW = 244,
        .outputSizeH = 244,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.00781f,
            .dtcPixelVarReciChn1 = 0.00781f,
            .dtcPixelVarReciChn2 = 0.00781f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 1,
            .leftPaddingSize = 10,
            .rightPaddingSize = 10,
            .topPaddingSize = 10,
            .bottomPaddingSize = 10,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_2)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_07.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_07.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 224, 244}, {1, 3, 224, 244}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 224,
        .inputSizeH = 224,
        .outputSizeW = 244,
        .outputSizeH = 224,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.00781f,
            .dtcPixelVarReciChn1 = 0.00781f,
            .dtcPixelVarReciChn2 = 0.00781f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 1,
            .leftPaddingSize = 20,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_3)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_08.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_08.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 256, 3}, {1, 256, 256, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 256, 276}, {1, 3, 256, 276}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_YUV_TO_RGB;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 1,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 256,
        .inputSizeH = 256,
        .outputSizeW = 276,
        .outputSizeH = 256,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 298,
            .cscMatrix01 = 0,
            .cscMatrix02 = 409,
            .cscMatrix10 = 298,
            .cscMatrix11 = -100,
            .cscMatrix12 = -208,
            .cscMatrix20 = 298,
            .cscMatrix21 = 516,
            .cscMatrix22 = 0,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 110,
            .inBias1 = 120,
            .inBias2 = 83
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 256,
            .cropSizeW = 256
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 123.675f,
            .dtcPixelMinChn1 = 116.28f,
            .dtcPixelMinChn2 = 103.53f,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.017124753f,
            .dtcPixelVarReciChn1 = 0.017507f,
            .dtcPixelVarReciChn2 = 0.0174292f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 1,
            .leftPaddingSize = 20,
            .rightPaddingSize = 0,
            .topPaddingSize = 0,
            .bottomPaddingSize = 0,
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_4)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_09.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_09.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 256, 1080, 3}, {1, 256, 1080, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 234, 244}, {1, 3, 234, 244}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 256,
        .inputSizeH = 1080,
        .outputSizeW = 244,
        .outputSizeH = 234,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 298,
            .cscMatrix01 = 10,
            .cscMatrix02 = 409,
            .cscMatrix10 = 298,
            .cscMatrix11 = -100,
            .cscMatrix12 = -208,
            .cscMatrix20 = 298,
            .cscMatrix21 = 516,
            .cscMatrix22 = 20,
            .outBias0 = 16,
            .outBias1 = 128,
            .outBias2 = 128,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 1,
            .cropStartPosH = 2,
            .cropStartPosW = 16,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 10,
            .dtcPixelMeanChn1 = 20,
            .dtcPixelMeanChn2 = 30,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 123.675f,
            .dtcPixelMinChn1 = 116.28f,
            .dtcPixelMinChn2 = 103.53f,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.017124753f,
            .dtcPixelVarReciChn1 = 0.017507004f,
            .dtcPixelVarReciChn2 = 0.017429193f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 1,
            .leftPaddingSize = 10,
            .rightPaddingSize = 10,
            .topPaddingSize = 5,
            .bottomPaddingSize = 5,
            .padValue = 10.5f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_5)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_10.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_10.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 288, 288}, {1, 3, 288, 288}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2,
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 224,
        .inputSizeH = 224,
        .outputSizeW = 288,
        .outputSizeH = 288,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0,
            .cropStartPosH = 0,
            .cropStartPosW = 0,
            .cropSizeH = 224,
            .cropSizeW = 224
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 128,
            .dtcPixelMeanChn1 = 128,
            .dtcPixelMeanChn2 = 128,
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0,
            .dtcPixelMinChn1 = 0,
            .dtcPixelMinChn2 = 0,
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 0.00781f,
            .dtcPixelVarReciChn1 = 0.00781f,
            .dtcPixelVarReciChn2 = 0.00781f,
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 1,
            .leftPaddingSize = 32,
            .rightPaddingSize = 32,
            .topPaddingSize = 32,
            .bottomPaddingSize = 32,
            .padValue = 255.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_6)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 100, 100, 3}, {1, 100, 100, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 100, 100}, {1, 3, 100, 100}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8", "padding":"aaa"})"))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_7)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 100, 100, 3}, {1, 100, 100, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 100, 100}, {1, 3, 100, 100}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8","padding":"true","left_padding_size":"33"})"))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_8)
{
    AippCompileInfo compileInfo = {56, 253952};
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 100, 100, 3}, {1, 100, 100, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 100, 100}, {1, 3, 100, 100}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
        Ops::Cv::AnyValue::CreateFrom<string>\
        (R"({"aipp_mode":"static","input_format":"RGB888_U8","padding":"true","left_padding_size":"10","padding_value":"65506"})"))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

TEST_F(AippTiling, aipp_tiling_padding_test_9)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_11.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_11.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 4096, 4096, 3}, {1, 4096, 4096, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 4096, 1144}, {1, 3, 4096, 1144}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {};  // 不检查
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_12: YUV420SP_U8转YUV444 - 不带swap
TEST_F(AippTiling, aipp_tiling_csc_test_12)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_12.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_12.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_YUV_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 1,  // YUV420SP_U8
        .outputFormat = 1,
        .batchNum = 1,
        .channelNum = 3,
        .inputSizeW = 1920,
        .inputSizeH = 1080,
        .outputSizeW = 1920,
        .outputSizeH = 1080,
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0,
            .rbuvSwapSwitch = 0,
            .axSwapSwitch = 0,
            .cscMatrix00 = 256,
            .cscMatrix01 = 0,
            .cscMatrix02 = 0,
            .cscMatrix10 = 0,
            .cscMatrix11 = 256,
            .cscMatrix12 = 0,
            .cscMatrix20 = 0,
            .cscMatrix21 = 0,
            .cscMatrix22 = 256,
            .outBias0 = 0,
            .outBias1 = 0,
            .outBias2 = 0,
            .inBias0 = 0,
            .inBias1 = 0,
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_13: YUV420SP_U8转YVU444 - 带rbuv swap (UV互换)
TEST_F(AippTiling, aipp_tiling_csc_test_13)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_13.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_13.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_YUV_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 1, .outputFormat = 1, .batchNum = 1, .channelNum = 3,
        .inputSizeW = 1920, .inputSizeH = 1080, .outputSizeW = 1920, .outputSizeH = 1080, .srcChannelOffset = 0,
        .cscParam = {.cscSwitch = 0, .rbuvSwapSwitch = 1, .axSwapSwitch = 0,
                     .cscMatrix00 = 256, .cscMatrix01 = 0, .cscMatrix02 = 0,
                     .cscMatrix10 = 0, .cscMatrix11 = 0, .cscMatrix12 = 256,
                     .cscMatrix20 = 0, .cscMatrix21 = 256, .cscMatrix22 = 0,
                     .outBias0 = 0, .outBias1 = 0, .outBias2 = 0, .inBias0 = 0, .inBias1 = 0, .inBias2 = 0},
        .cropParam = {.cropSwitch = 0, .cropStartPosH = 0, .cropStartPosW = 0, .cropSizeH = 1080, .cropSizeW = 1920},
        .dtcParam = {.dtcPixelMeanChn0 = 0, .dtcPixelMeanChn1 = 0, .dtcPixelMeanChn2 = 0, .dtcPixelMeanChn3 = 0,
                     .dtcPixelMinChn0 = 0, .dtcPixelMinChn1 = 0, .dtcPixelMinChn2 = 0, .dtcPixelMinChn3 = 0,
                     .dtcPixelVarReciChn0 = 1, .dtcPixelVarReciChn1 = 1, .dtcPixelVarReciChn2 = 1, .dtcPixelVarReciChn3 = 1},
        .paddingParam = {.paddingSwitch = 0, .leftPaddingSize = 0, .rightPaddingSize = 0, .topPaddingSize = 0, .bottomPaddingSize = 0, .padValue = 0.0f}
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_14: RGB888_U8转RGB - 不带csc和swap (直通模式)
TEST_F(AippTiling, aipp_tiling_csc_test_14)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_14.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_14.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 3,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 256, 
            .cscMatrix01 = 0, 
            .cscMatrix02 = 0,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 256, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 256,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_15: RGB888_U8转YVU444 - 带csc矩阵转换(BT601标准YUV转换)
TEST_F(AippTiling, aipp_tiling_csc_test_15)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_15.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_15.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 3,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 77, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 29,
            .cscMatrix10 = 128, 
            .cscMatrix11 = -107, 
            .cscMatrix12 = -21,
            .cscMatrix20 = -43, 
            .cscMatrix21 = -85, 
            .cscMatrix22 = 128,
            .outBias0 = 0, 
            .outBias1 = 128, 
            .outBias2 = 128, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0},
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920},
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_16: BGR888_U8转RGB - 带rbuv swap (BGR通道互换)
TEST_F(AippTiling, aipp_tiling_csc_test_16)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_16.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_16.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 3,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0, 
            .rbuvSwapSwitch = 1, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 0, 
            .cscMatrix01 = 0, 
            .cscMatrix02 = 256,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 256, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 256, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 0,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_17: BGR888_U8转BGR - 不带swap (直通)
TEST_F(AippTiling, aipp_tiling_csc_test_17)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_17.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_17.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 3}, {1, 1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_PASS_THROUGH;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 2, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 3,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 256, 
            .cscMatrix01 = 0, 
            .cscMatrix02 = 0,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 256, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 256,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_18: XRGB8888_U8转YUV444 - 带ax swap和csc矩阵
TEST_F(AippTiling, aipp_tiling_csc_test_18)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_18.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_18.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 1,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 1,
            .cscMatrix00 = 77, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 29,
            .cscMatrix10 = -43, 
            .cscMatrix11 = -85, 
            .cscMatrix12 = 128,
            .cscMatrix20 = 128, 
            .cscMatrix21 = -107, 
            .cscMatrix22 = -21,
            .outBias0 = 0, 
            .outBias1 = 128, 
            .outBias2 = 128, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_19: XRGB8888_U8转YVU444 - 带ax swap和csc矩阵(UV互换)
TEST_F(AippTiling, aipp_tiling_csc_test_19)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_19.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_19.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 3, 1080, 1920}, {1, 3, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_YUV;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 1,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 1,
            .cscMatrix00 = 77, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 29,
            .cscMatrix10 = 128, 
            .cscMatrix11 = -107, 
            .cscMatrix12 = -21,
            .cscMatrix20 = -43, 
            .cscMatrix21 = -85, 
            .cscMatrix22 = 128,
            .outBias0 = 0, 
            .outBias1 = 128, 
            .outBias2 = 128, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_20: XRGB8888_U8转GRAY - 带ax swap和灰度矩阵
TEST_F(AippTiling, aipp_tiling_csc_test_20)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_20.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_20.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 1, 1080, 1920}, {1, 1, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_GRAY;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 1,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 1,
            .cscMatrix00 = 76, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 30,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 0, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 0,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_21: XBGR8888_U8转GRAY - 带ax swap和rbuv swap
TEST_F(AippTiling, aipp_tiling_csc_test_21)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_21.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_21.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 1, 1080, 1920}, {1, 1, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_GRAY;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 1,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 1, 
            .axSwapSwitch = 1,
            .cscMatrix00 = 30, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 76,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 0, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 0,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_22: RGBX8888_U8转GRAY - 带ax swap关闭
TEST_F(AippTiling, aipp_tiling_csc_test_22)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_22.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_22.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 1, 1080, 1920}, {1, 1, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_GRAY;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 76, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 30,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 0, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 0,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_23: BGRX8888_U8转GRAY - 带rbuv swap和ax swap关闭
TEST_F(AippTiling, aipp_tiling_csc_test_23)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_23.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_23.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 4}, {1, 1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 1, 1080, 1920}, {1, 1, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_RGB_TO_GRAY;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 3, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 4,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 1, 
            .rbuvSwapSwitch = 1, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 30, 
            .cscMatrix01 = 150, 
            .cscMatrix02 = 76,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 0, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 0,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}

// test_24: YUV400_U8转GRAY - 不带csc (YUV400单通道直通)
TEST_F(AippTiling, aipp_tiling_csc_test_24)
{
    AippCompileInfo compileInfo = {56, 253952};
    std::string filePath = std::filesystem::current_path() / "aipp_ut_test_24.cfg";
    std::string command = "cp ../../../../image/aipp/tests/ut/op_host/aipp_ut_test_24.cfg " + filePath;
    system(command.c_str());
    gert::TilingContextPara tilingContextPara("Aipp",
        {{{{1, 1080, 1920, 1}, {1, 1080, 1920, 1}}, ge::DT_UINT8, ge::FORMAT_NHWC}},
        {{{{1, 1, 1080, 1920}, {1, 1, 1080, 1920}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(filePath))},
        &compileInfo);

    int64_t expectTilingKey = optiling::AIPP_YUV_TO_GRAY;
    Aipp_Kernel::AippTilingData expectTiling = {
        .imageFormat = 4, 
        .outputFormat = 1, 
        .batchNum = 1, 
        .channelNum = 1,
        .inputSizeW = 1920, 
        .inputSizeH = 1080, 
        .outputSizeW = 1920, 
        .outputSizeH = 1080, 
        .srcChannelOffset = 0,
        .cscParam = {
            .cscSwitch = 0, 
            .rbuvSwapSwitch = 0, 
            .axSwapSwitch = 0,
            .cscMatrix00 = 256, 
            .cscMatrix01 = 0, 
            .cscMatrix02 = 0,
            .cscMatrix10 = 0, 
            .cscMatrix11 = 256, 
            .cscMatrix12 = 0,
            .cscMatrix20 = 0, 
            .cscMatrix21 = 0, 
            .cscMatrix22 = 256,
            .outBias0 = 0, 
            .outBias1 = 0, 
            .outBias2 = 0, 
            .inBias0 = 0, 
            .inBias1 = 0, 
            .inBias2 = 0
        },
        .cropParam = {
            .cropSwitch = 0, 
            .cropStartPosH = 0, 
            .cropStartPosW = 0, 
            .cropSizeH = 1080, 
            .cropSizeW = 1920
        },
        .dtcParam = {
            .dtcPixelMeanChn0 = 0, 
            .dtcPixelMeanChn1 = 0, 
            .dtcPixelMeanChn2 = 0, 
            .dtcPixelMeanChn3 = 0,
            .dtcPixelMinChn0 = 0, 
            .dtcPixelMinChn1 = 0, 
            .dtcPixelMinChn2 = 0, 
            .dtcPixelMinChn3 = 0,
            .dtcPixelVarReciChn0 = 1, 
            .dtcPixelVarReciChn1 = 1, 
            .dtcPixelVarReciChn2 = 1, 
            .dtcPixelVarReciChn3 = 1
        },
        .paddingParam = {
            .paddingSwitch = 0, 
            .leftPaddingSize = 0, 
            .rightPaddingSize = 0, 
            .topPaddingSize = 0, 
            .bottomPaddingSize = 0, 
            .padValue = 0.0f
        }
    };
    std::vector<size_t> expectWorkspaces = {16777216};
    AippExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTiling, expectWorkspaces);
}