/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ctime>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ops_proto_legacy.h"
#include "ge_ir_build.h"

#include "../op_graph/crop_and_resize_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_DOUBLE || dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        dilation = twoByte;
    }
    // 其余（uint8/int8 等）按 1 字节
    return dilation;
}

// ==================== check_supported 引擎路由用例 ====================
// 用例配置：输入定义 + 期望输出 shape；expectAiCpuFallback 仅用于日志标注
// （判定标准为图编译+执行成功且输出 shape 正确：违规输入需 check 拒绝后 AiCpu
// fallback 才能执行成功，合规输入由 AiCore 直行；引擎归属证据由 plog 复核）
struct CaseConfig {
    string name;
    DataType xDtype;
    vector<int64_t> xShape;
    vector<int64_t> boxesShape; // {num_boxes, 4}
    vector<int64_t> cropSize;   // const 值
    vector<int64_t> expectOutShape;
    bool expectAiCpuFallback;
};

int32_t GenTensorByDesc(const TensorDesc& desc, int64_t elemNum, Tensor& tensor, uint8_t fillValue)
{
    // 用 vector 持有缓冲直至函数返回，避免依赖 Tensor 构造的拷贝语义
    // （若框架实现为浅持有，函数返回后pData即悬垂）
    size_t byteSize = static_cast<size_t>(elemNum) * GetDataTypeSize(desc.GetDataType());
    std::vector<uint8_t> buffer(byteSize, fillValue);
    tensor = Tensor(desc, buffer.data(), byteSize);
    return SUCCESS;
}

int32_t BuildCaseGraph(const CaseConfig& cs, Graph& graph, std::vector<ge::Tensor>& inputTensors,
                       std::vector<Operator>& inputs)
{
    Status ret = SUCCESS;
    auto opNode = op::CropAndResize("crop_and_resize_" + cs.name);

    // x: Data 输入
    TensorDesc xDesc = TensorDesc(ge::Shape(cs.xShape), FORMAT_NHWC, cs.xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    auto xData = op::Data("x_" + cs.name).set_attr_index(0);
    Tensor xTensor;
    int64_t xSize = 1;
    for (auto d : cs.xShape) {
        xSize *= d;
    }
    ret = GenTensorByDesc(xDesc, xSize, xTensor, 1);
    if (ret != SUCCESS) {
        return FAILED;
    }
    inputTensors.push_back(xTensor);
    graph.AddOp(xData);
    opNode.set_input_x(xData);
    opNode.update_input_desc_x(xDesc);
    inputs.push_back(xData);

    // boxes: Data 输入
    TensorDesc boxesDesc = TensorDesc(ge::Shape(cs.boxesShape), FORMAT_ND, DT_FLOAT);
    boxesDesc.SetPlacement(ge::kPlacementHost);
    auto boxesData = op::Data("boxes_" + cs.name).set_attr_index(1);
    Tensor boxesTensor;
    ret = GenTensorByDesc(boxesDesc, cs.boxesShape[0] * cs.boxesShape[1], boxesTensor, 0x66);
    if (ret != SUCCESS) {
        return FAILED;
    }
    inputTensors.push_back(boxesTensor);
    graph.AddOp(boxesData);
    opNode.set_input_boxes(boxesData);
    opNode.update_input_desc_boxes(boxesDesc);
    inputs.push_back(boxesData);

    // box_index: Data 输入
    TensorDesc boxIdxDesc = TensorDesc(ge::Shape({cs.boxesShape[0]}), FORMAT_ND, DT_INT32);
    boxIdxDesc.SetPlacement(ge::kPlacementHost);
    auto boxIdxData = op::Data("box_index_" + cs.name).set_attr_index(2);
    Tensor boxIdxTensor;
    ret = GenTensorByDesc(boxIdxDesc, cs.boxesShape[0], boxIdxTensor, 0);
    if (ret != SUCCESS) {
        return FAILED;
    }
    inputTensors.push_back(boxIdxTensor);
    graph.AddOp(boxIdxData);
    opNode.set_input_box_index(boxIdxData);
    opNode.update_input_desc_box_index(boxIdxDesc);
    inputs.push_back(boxIdxData);

    // crop_size: Const 输入（值进 weights，check 阶段可读）
    TensorDesc cropSizeDesc = TensorDesc(ge::Shape({2}), FORMAT_ND, DT_INT32);
    auto cropSizeConst = op::Const("crop_size_" + cs.name);
    int32_t cropVals[2] = {static_cast<int32_t>(cs.cropSize[0]), static_cast<int32_t>(cs.cropSize[1])};
    Tensor cropSizeTensor = Tensor(cropSizeDesc, reinterpret_cast<uint8_t*>(cropVals), sizeof(cropVals));
    cropSizeConst.SetAttr("value", cropSizeTensor);
    cropSizeConst.update_output_desc_y(cropSizeDesc);
    graph.AddOp(cropSizeConst);
    opNode.set_input_crop_size(cropSizeConst);
    opNode.update_input_desc_crop_size(cropSizeDesc);

    // 输出 desc
    TensorDesc yDesc = TensorDesc(ge::Shape(cs.expectOutShape), FORMAT_NHWC, DT_FLOAT);
    opNode.update_output_desc_y(yDesc);

    graph.SetInputs(inputs).SetOutputs({opNode});
    return SUCCESS;
}

// 原单用例流程（uint8 冒烟）保留为用例之一：与其余用例共用 BuildCaseGraph
int RunOneCase(ge::Session* session, uint32_t graphId, const CaseConfig& cs)
{
    string graphName = "check_case_" + cs.name;
    Graph graph(graphName.c_str());
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> inputs;
    if (BuildCaseGraph(cs, graph, inputTensors, inputs) != SUCCESS) {
        printf("%s - ERROR - [%s]: build graph failed\n", GetTime().c_str(), cs.name.c_str());
        return FAILED;
    }
    std::map<AscendString, AscendString> graphOptions = {};
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("%s - ERROR - [%s]: AddGraph failed\n", GetTime().c_str(), cs.name.c_str());
        return FAILED;
    }
    std::vector<ge::Tensor> output;
    if (session->RunGraph(graphId, inputTensors, output) != SUCCESS) {
        printf("%s - ERROR - [%s]: RunGraph failed\n", GetTime().c_str(), cs.name.c_str());
        return FAILED;
    }
    if (output.empty()) {
        printf("%s - ERROR - [%s]: empty output\n", GetTime().c_str(), cs.name.c_str());
        return FAILED;
    }
    auto gotShape = output[0].GetTensorDesc().GetShape().GetDims();
    if (gotShape.size() != cs.expectOutShape.size()) {
        printf("%s - ERROR - [%s]: output rank %zu != expect %zu\n", GetTime().c_str(), cs.name.c_str(),
               gotShape.size(), cs.expectOutShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < gotShape.size(); i++) {
        if (gotShape[i] != cs.expectOutShape[i]) {
            printf("%s - ERROR - [%s]: output dim[%zu]=%ld != expect %ld\n", GetTime().c_str(), cs.name.c_str(), i,
                   (long)gotShape[i], (long)cs.expectOutShape[i]);
            return FAILED;
        }
    }
    printf("%s - INFO - [%s]: PASS (%s)\n", GetTime().c_str(), cs.name.c_str(),
           cs.expectAiCpuFallback ? "expect fallback AiCpu" : "expect AiCore");
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - INFO - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    // 用例集：原 uint8 冒烟用例（AiCpu 路由）+ check 引擎路由专项
    std::vector<CaseConfig> cases = {
        // 原冒烟用例：uint8 x（AiCpu 承接）
        {"smoke_uint8", DT_UINT8, {2, 2, 2, 2}, {2, 4}, {2, 2}, {2, 2, 2, 2}, true},
        // 违规组：AiCore 约束不满足，需 fallback AiCpu 才能执行成功
        {"x_dtype_double", DT_DOUBLE, {2, 8, 8, 64}, {64, 4}, {4, 4}, {64, 4, 4, 64}, true},
        {"depth_lt_256", DT_FLOAT, {2, 128, 128, 64}, {64, 4}, {8, 8}, {64, 8, 8, 64}, true},
        {"num_boxes_lt_50", DT_FLOAT, {2, 128, 128, 256}, {32, 4}, {8, 8}, {32, 8, 8, 256}, true},
        {"crop_gt_16", DT_FLOAT, {2, 128, 128, 256}, {64, 4}, {17, 17}, {64, 17, 17, 256}, true},
        {"fp32_hw_gt_32765", DT_FLOAT, {2, 181, 182, 256}, {64, 4}, {8, 8}, {64, 8, 8, 256}, true},
        // 合规组：AiCore 直行（含边界值防误伤）
        {"valid_fp32", DT_FLOAT, {2, 128, 128, 256}, {64, 4}, {8, 8}, {64, 8, 8, 256}, false},
        {"boundary_crop_16", DT_FLOAT, {2, 128, 128, 256}, {64, 4}, {16, 16}, {64, 16, 16, 256}, false},
        {"boundary_num_boxes_51", DT_FLOAT, {2, 128, 128, 256}, {51, 4}, {8, 8}, {51, 8, 8, 256}, false},
    };

    int failed = 0;
    uint32_t graphId = 0;
    for (const auto& cs : cases) {
        if (RunOneCase(session, graphId++, cs) != SUCCESS) {
            failed++;
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    if (failed != 0) {
        printf("%s - ERROR - [CHECK]: %d case(s) FAILED\n", GetTime().c_str(), failed);
        return FAILED;
    }
    printf("%s - INFO - [CHECK]: all %zu cases PASSED\n", GetTime().c_str(), cases.size());
    return SUCCESS;
}
