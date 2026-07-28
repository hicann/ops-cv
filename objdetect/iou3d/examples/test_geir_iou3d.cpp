/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ---------------------------------------------------------------------------------------------------------
 * Iou3D 算子 GE IR 图模式调用示例。
 *
 * 构图：Data(bboxes[B,7,N]) + Data(gtboxes[B,7,K]) -> Iou3D -> iou[B,N,K]
 *   通过 op::Iou3D（op_graph/iou3d_proto.h 注册的原型）建图，交给 ge::Session 编译执行。
 *   输入用 float32 常量数据（单位框 + 平移框），输出 dump 为 bin 供离线核对。
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
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

#include "../op_graph/iou3d_proto.h"

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

// 构造 float32 host Tensor
static Tensor MakeFloatTensor(const vector<int64_t>& shape, const vector<float>& data)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    Tensor t(desc);
    t.SetData(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float));
    return t;
}

static int32_t WriteBin(const string& path, const uint8_t* data, size_t size)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr)
        return FAILED;
    fwrite(data, sizeof(uint8_t), size, fp);
    fclose(fp);
    return SUCCESS;
}

int main()
{
    // B=1, N=2, K=3 —— 与 aclnn 示例同几何
    const int64_t B = 1, N = 2, K = 3;
    // bboxes[B,7,N]：DoF 顺序 x,y,z,w,h,d,theta，每 DoF 连续 N 个框
    vector<float> bboxesData = {
        0.0f, 5.0f, // x
        0.0f, 5.0f, // y
        0.0f, 5.0f, // z
        1.0f, 1.0f, // w
        1.0f, 1.0f, // h
        1.0f, 1.0f, // d
        0.0f, 0.0f, // theta
    };
    // gtboxes[B,7,K]
    vector<float> gtboxesData = {
        0.0f, 0.5f, 0.0f,          // x
        0.0f, 0.0f, 0.0f,          // y
        0.0f, 0.0f, 0.0f,          // z
        1.0f, 1.0f, 1.0f,          // w
        1.0f, 1.0f, 1.0f,          // h
        1.0f, 1.0f, 1.0f,          // d
        0.0f, 0.0f, 0.7853981634f, // theta (pi/4)
    };
    vector<int64_t> bboxesShape = {B, 7, N};
    vector<int64_t> gtboxesShape = {B, 7, K};
    vector<int64_t> iouShape = {B, N, K};

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

    // 构图
    Graph graph("iou3d_geir_graph");

    auto bboxesData0 = op::Data("bboxes").set_attr_index(0);
    TensorDesc bboxesDesc(ge::Shape(bboxesShape), FORMAT_ND, DT_FLOAT);
    bboxesData0.update_input_desc_x(bboxesDesc);
    bboxesData0.update_output_desc_y(bboxesDesc);

    auto gtboxesData0 = op::Data("gtboxes").set_attr_index(1);
    TensorDesc gtboxesDesc(ge::Shape(gtboxesShape), FORMAT_ND, DT_FLOAT);
    gtboxesData0.update_input_desc_x(gtboxesDesc);
    gtboxesData0.update_output_desc_y(gtboxesDesc);

    auto iou3d = op::Iou3D("iou3d_0");
    iou3d.set_input_bboxes(bboxesData0);
    iou3d.set_input_gtboxes(gtboxesData0);
    iou3d.update_input_desc_bboxes(bboxesDesc);
    iou3d.update_input_desc_gtboxes(gtboxesDesc);
    TensorDesc iouDesc(ge::Shape(iouShape), FORMAT_ND, DT_FLOAT);
    iou3d.update_output_desc_iou(iouDesc);

    std::vector<Operator> inputs;
    inputs.push_back(bboxesData0);
    inputs.push_back(gtboxesData0);
    std::vector<Operator> outputs;
    outputs.push_back(iou3d);
    graph.SetInputs(inputs).SetOutputs(outputs);

    // Session
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

    // 输入
    std::vector<Tensor> inputTensors;
    inputTensors.push_back(MakeFloatTensor(bboxesShape, bboxesData));
    inputTensors.push_back(MakeFloatTensor(gtboxesShape, gtboxesData));

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
        uint8_t* p = outputTensors[i].GetData();
        if (p == nullptr) {
            printf("%s - ERROR - [GEIR]: output[%zu] GetData returned null\n", GetTime().c_str(), i);
            continue;
        }
        size_t sz = outputTensors[i].GetSize();
        int64_t cnt = static_cast<int64_t>(sz / sizeof(float));
        printf("%s - INFO - [GEIR]: output[%zu] size=%zu (%ld floats)\n", GetTime().c_str(), i, sz, cnt);
        const float* fp = reinterpret_cast<const float*>(p);
        for (int64_t e = 0; e < cnt; ++e) {
            printf("  iou[%ld] = %.6f\n", e, fp[e]);
        }
        WriteBin("./iou3d_geir_output_" + std::to_string(i) + ".bin", p, sz);
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
