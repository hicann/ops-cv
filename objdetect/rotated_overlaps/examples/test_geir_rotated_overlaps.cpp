/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ---------------------------------------------------------------------------------------------------------
 * RotatedOverlaps 算子 GE IR 图模式调用示例。
 *
 * 构图：Data(boxes[B,5,N]) + Data(query_boxes[B,5,K]) -> RotatedOverlaps -> overlaps[B,N,K]
 *   通过 op::RotatedOverlaps（op_graph/rotated_overlaps_proto.h 注册的原型）建图，交给 ge::Session 编译执行。
 *   输入使用 float32 的 xywht 旋转框数据，输出 dump 为 bin 供离线核对。
 */

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h" // op::Data

#include "../op_graph/rotated_overlaps_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return tmp;
}

static Tensor MakeFloatTensor(const vector<int64_t>& shape, const vector<float>& data)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    Tensor tensor(desc);
    tensor.SetData(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float));
    return tensor;
}

static int32_t WriteBin(const string& path, const uint8_t* data, size_t size)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(data, sizeof(uint8_t), size, fp);
    fclose(fp);
    return SUCCESS;
}

int main()
{
    // B=1, N=2, K=3；框格式为 [x, y, w, h, theta]，theta 单位为度。
    const int64_t batch = 1;
    const int64_t numBoxes = 2;
    const int64_t numQueries = 3;
    const vector<float> boxesData = {
        0.0f, 5.0f, // x
        0.0f, 5.0f, // y
        2.0f, 2.0f, // w
        2.0f, 2.0f, // h
        0.0f, 0.0f, // theta
    };
    const vector<float> queryBoxesData = {
        0.0f, 0.5f, 5.0f,  // x
        0.0f, 0.0f, 5.0f,  // y
        2.0f, 2.0f, 2.0f,  // w
        2.0f, 2.0f, 2.0f,  // h
        0.0f, 0.0f, 45.0f, // theta
    };
    const vector<int64_t> boxesShape = {batch, 5, numBoxes};
    const vector<int64_t> queryBoxesShape = {batch, 5, numQueries};
    const vector<int64_t> overlapsShape = {batch, numBoxes, numQueries};

    printf("%s - INFO - [GEIR]: GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    Graph graph("rotated_overlaps_geir_graph");
    auto boxesDataOp = op::Data("boxes").set_attr_index(0);
    TensorDesc boxesDesc(ge::Shape(boxesShape), FORMAT_ND, DT_FLOAT);
    boxesDataOp.update_input_desc_x(boxesDesc);
    boxesDataOp.update_output_desc_y(boxesDesc);

    auto queryBoxesDataOp = op::Data("query_boxes").set_attr_index(1);
    TensorDesc queryBoxesDesc(ge::Shape(queryBoxesShape), FORMAT_ND, DT_FLOAT);
    queryBoxesDataOp.update_input_desc_x(queryBoxesDesc);
    queryBoxesDataOp.update_output_desc_y(queryBoxesDesc);

    auto rotatedOverlaps = op::RotatedOverlaps("rotated_overlaps_0");
    rotatedOverlaps.set_input_boxes(boxesDataOp);
    rotatedOverlaps.set_input_query_boxes(queryBoxesDataOp);
    rotatedOverlaps.set_attr_trans(false);
    rotatedOverlaps.update_input_desc_boxes(boxesDesc);
    rotatedOverlaps.update_input_desc_query_boxes(queryBoxesDesc);
    TensorDesc overlapsDesc(ge::Shape(overlapsShape), FORMAT_ND, DT_FLOAT);
    rotatedOverlaps.update_output_desc_overlaps(overlapsDesc);

    std::vector<Operator> inputs;
    inputs.push_back(boxesDataOp);
    inputs.push_back(queryBoxesDataOp);
    std::vector<Operator> outputs;
    outputs.push_back(rotatedOverlaps);
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions;
    printf("%s - INFO - [GEIR]: create Session\n", GetTime().c_str());
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [GEIR]: create Session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graphOptions;
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: AddGraph success\n", GetTime().c_str());

    std::vector<Tensor> inputTensors;
    inputTensors.push_back(MakeFloatTensor(boxesShape, boxesData));
    inputTensors.push_back(MakeFloatTensor(queryBoxesShape, queryBoxesData));

    printf("%s - INFO - [GEIR]: RunGraph\n", GetTime().c_str());
    std::vector<Tensor> outputTensors;
    ret = session->RunGraph(graphId, inputTensors, outputTensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [GEIR]: %s\n", GetTime().c_str(), errMsg.GetString());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: RunGraph success, output_num=%zu\n", GetTime().c_str(), outputTensors.size());

    for (size_t i = 0; i < outputTensors.size(); ++i) {
        uint8_t* data = outputTensors[i].GetData();
        if (data == nullptr) {
            printf("%s - ERROR - [GEIR]: output[%zu] GetData returned null\n", GetTime().c_str(), i);
            continue;
        }
        const size_t size = outputTensors[i].GetSize();
        const int64_t count = static_cast<int64_t>(size / sizeof(float));
        printf("%s - INFO - [GEIR]: output[%zu] size=%zu (%ld floats)\n", GetTime().c_str(), i, size, count);
        const float* values = reinterpret_cast<const float*>(data);
        for (int64_t index = 0; index < count; ++index) {
            printf("  overlaps[%ld] = %.6f\n", index, values[index]);
        }
        WriteBin("./rotated_overlaps_geir_output_" + std::to_string(i) + ".bin", data, size);
    }

    printf("%s - INFO - [GEIR]: GEFinalize\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: done\n", GetTime().c_str());
    return SUCCESS;
}
