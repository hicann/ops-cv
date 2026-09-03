/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_grid_unnormal.cpp
 * \brief GridUnnormal 算子 GE IR 图模式调用样例。
 *
 * 样例配置（align_corners = false）：
 *   grid   : DT_FLOAT, shape = [1, 6, 5, 2]，每元素填 0.3f
 *   assist : DT_FLOAT, shape = [1, 6, 5, 2]，每元素填 5.0f
 * 期望：
 *   t        = (0.3 + 1) * 0.5 = 0.65
 *   pos_base = 0.65 * 5 - 0.5  = 2.75
 *   position = floor(2.75)     = 2   (int32)
 *   diff     = 2.75 - 2        = 0.75
 */

#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../../op_graph/grid_unnormal_proto.h"

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
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return 1;
}

static int32_t GenConstFloatData(const vector<int64_t>& shapes, Tensor& tensor, TensorDesc& desc, float value,
                                 vector<std::unique_ptr<float[]>>& inputData)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    std::unique_ptr<float[]> pData(new (std::nothrow) float[size]);
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData.get()), size * sizeof(float));
    inputData.push_back(std::move(pData));
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph, vector<std::unique_ptr<float[]>>& inputData)
{
    Status ret = SUCCESS;
    auto gridUnnormal = op::GridUnnormal("gridUnnormal_1");

    std::vector<int64_t> shape = {1, 6, 5, 2};

    // 输入 grid（placeholder，index 0），填 0.3f
    auto grid = op::Data("grid").set_attr_index(0);
    TensorDesc gridDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    gridDesc.SetPlacement(ge::kPlacementHost);
    Tensor gridTensor;
    ret = GenConstFloatData(shape, gridTensor, gridDesc, 0.3f, inputData);
    if (ret != SUCCESS) {
        return FAILED;
    }
    grid.update_input_desc_x(gridDesc);
    grid.update_output_desc_y(gridDesc);
    input.push_back(gridTensor);
    graph.AddOp(grid);
    gridUnnormal.set_input_grid(grid);
    inputs.push_back(grid);

    // 输入 assist（placeholder，index 1），填 5.0f
    auto assist = op::Data("assist").set_attr_index(1);
    TensorDesc assistDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    assistDesc.SetPlacement(ge::kPlacementHost);
    Tensor assistTensor;
    ret = GenConstFloatData(shape, assistTensor, assistDesc, 5.0f, inputData);
    if (ret != SUCCESS) {
        return FAILED;
    }
    assist.update_input_desc_x(assistDesc);
    assist.update_output_desc_y(assistDesc);
    input.push_back(assistTensor);
    graph.AddOp(assist);
    gridUnnormal.set_input_assist(assist);
    inputs.push_back(assist);

    // 属性
    gridUnnormal.set_attr_align_corners(false);

    // 输出描述
    TensorDesc diffDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    TensorDesc posDesc(ge::Shape(shape), FORMAT_ND, DT_INT32);
    gridUnnormal.update_output_desc_diff(diffDesc);
    gridUnnormal.update_output_desc_position(posDesc);

    outputs.push_back(gridUnnormal);
    return SUCCESS;
}

static void PrintOutputs(const std::vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < output.size(); i++) {
        DataType dt = output[i].GetTensorDesc().GetDataType();
        int64_t n = output[i].GetTensorDesc().GetShape().GetShapeSize();
        const uint8_t* data = output[i].GetData();
        printf("output[%zu] dtype=%d shapeSize=%ld first-elems: ", i, static_cast<int>(dt), n);
        int64_t show = n < 4 ? n : 4;
        if (dt == ge::DT_FLOAT) {
            const float* p = reinterpret_cast<const float*>(data);
            for (int64_t j = 0; j < show; j++) {
                printf("%f ", p[j]);
            }
        } else if (dt == ge::DT_INT32) {
            const int32_t* p = reinterpret_cast<const int32_t*>(data);
            for (int64_t j = 0; j < show; j++) {
                printf("%d ", p[j]);
            }
        }
        printf("\n");
    }
}

static Status BuildAndRunGraph()
{
    Graph graph("gridunnormal_ge_ir_test");
    std::vector<ge::Tensor> input;
    vector<std::unique_ptr<float[]>> inputData;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    Status ret = CreateOppInGraph(input, inputs, outputs, graph, inputData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> buildOptions = {};
    std::unique_ptr<Session> session(new (std::nothrow) Session(buildOptions));
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success, outputs=%zu\n", GetTime().c_str(), output.size());
    PrintOutputs(output);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    ret = BuildAndRunGraph();
    Status finalizeRet = ge::GEFinalize();
    if (ret != SUCCESS) {
        return FAILED;
    }
    if (finalizeRet != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}
