/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "array_ops.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sorted_nms_proto.h"

namespace {
constexpr int32_t kSuccess = 0;
constexpr int32_t kFailed = -1;
constexpr int64_t kBoxCount = 5;
constexpr size_t kExpectedCount = 3;

ge::TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    return desc;
}

template <typename T>
ge::Tensor MakeTensor(const ge::TensorDesc& desc, std::vector<T>& data)
{
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(T));
}

bool CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    const std::array<int32_t, kExpectedCount> expected = {0, 2, 3};
    const std::vector<int64_t> expectedShape = {static_cast<int64_t>(kExpectedCount)};
    if (outputs.size() != 1 || outputs[0].GetTensorDesc().GetDataType() != ge::DT_INT32 ||
        outputs[0].GetTensorDesc().GetShape().GetDims() != expectedShape) {
        std::cerr << "Unexpected SortedNMS output descriptor: output_count=" << outputs.size();
        if (!outputs.empty()) {
            const auto& outputDesc = outputs[0].GetTensorDesc();
            std::cerr << ", dtype=" << static_cast<int32_t>(outputDesc.GetDataType()) << ", shape=[";
            const auto& dims = outputDesc.GetShape().GetDims();
            for (size_t i = 0; i < dims.size(); ++i) {
                std::cerr << (i == 0 ? "" : ", ") << dims[i];
            }
            std::cerr << "], bytes=" << outputs[0].GetSize();
            const uint8_t* outputData = outputs[0].GetData();
            const size_t valueCount = outputs[0].GetSize() / sizeof(int32_t);
            if (outputData != nullptr && valueCount > 0) {
                const size_t printCount = valueCount < kExpectedCount ? valueCount : kExpectedCount;
                std::cerr << ", raw_values=[";
                for (size_t i = 0; i < printCount; ++i) {
                    int32_t value = 0;
                    std::memcpy(&value, outputData + i * sizeof(value), sizeof(value));
                    std::cerr << (i == 0 ? "" : ", ") << value;
                }
                std::cerr << "]";
            }
        }
        std::cerr << std::endl;
        return false;
    }

    const uint8_t* outputData = outputs[0].GetData();
    if (outputData == nullptr) {
        std::cerr << "SortedNMS output data is null" << std::endl;
        return false;
    }

    std::array<int32_t, kExpectedCount> actual{};
    std::memcpy(actual.data(), outputData, sizeof(actual));
    for (size_t i = 0; i < actual.size(); ++i) {
        std::cout << "selected_indices[" << i << "] = " << actual[i] << std::endl;
        if (actual[i] != expected[i]) {
            std::cerr << "Unexpected SortedNMS result at " << i << ": expected " << expected[i] << ", got " << actual[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                        {"ge.graphRunMode", "1"}};
    ge::Status ret = ge::GEInitialize(globalOptions);
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "GEInitialize failed: " << ret << std::endl;
        return kFailed;
    }

    int32_t result = kFailed;
    {
        ge::Graph graph("sorted_nms_geir_example");
        auto boxes = ge::op::Data("boxes").set_attr_index(0);
        auto sortedScores = ge::op::Data("sorted_scores").set_attr_index(1);
        auto inputIndices = ge::op::Data("input_indices").set_attr_index(2);
        auto maxOutputSize = ge::op::Data("max_output_size").set_attr_index(3);
        auto iouThreshold = ge::op::Data("iou_threshold").set_attr_index(4);
        auto scoreThreshold = ge::op::Data("score_threshold").set_attr_index(5);
        auto sortedNms = ge::op::SortedNMS("sorted_nms");

        const ge::TensorDesc boxesDesc = MakeTensorDesc({kBoxCount, 4}, ge::DT_FLOAT);
        const ge::TensorDesc scoresDesc = MakeTensorDesc({kBoxCount}, ge::DT_FLOAT);
        const ge::TensorDesc indicesDesc = MakeTensorDesc({kBoxCount}, ge::DT_INT32);
        // GEIR graph inputs use the supported single-element scalar representation.
        const ge::TensorDesc scalarFloatDesc = MakeTensorDesc({1}, ge::DT_FLOAT);
        const ge::TensorDesc scalarIntDesc = MakeTensorDesc({1}, ge::DT_INT32);
        const ge::TensorDesc outputDesc = MakeTensorDesc({ge::UNKNOWN_DIM}, ge::DT_INT32);

        boxes.update_input_desc_x(boxesDesc);
        sortedScores.update_input_desc_x(scoresDesc);
        inputIndices.update_input_desc_x(indicesDesc);
        maxOutputSize.update_input_desc_x(scalarIntDesc);
        iouThreshold.update_input_desc_x(scalarFloatDesc);
        scoreThreshold.update_input_desc_x(scalarFloatDesc);
        sortedNms.set_input_boxes(boxes)
            .set_input_sorted_scores(sortedScores)
            .set_input_input_indices(inputIndices)
            .set_input_max_output_size(maxOutputSize)
            .set_input_iou_threshold(iouThreshold)
            .set_input_score_threshold(scoreThreshold)
            .set_attr_offset(0);
        sortedNms.update_output_desc_selected_indices(outputDesc);

        graph.AddOp(boxes);
        graph.AddOp(sortedScores);
        graph.AddOp(inputIndices);
        graph.AddOp(maxOutputSize);
        graph.AddOp(iouThreshold);
        graph.AddOp(scoreThreshold);
        const std::vector<ge::Operator> graphInputs = {boxes,         sortedScores, inputIndices,
                                                       maxOutputSize, iouThreshold, scoreThreshold};
        const std::vector<ge::Operator> graphOutputs = {sortedNms};
        graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

        // Scores are descending. Box 1 overlaps box 0 and must be suppressed.
        std::vector<float> boxesData = {0.0F,  0.0F,  10.0F, 10.0F, 1.0F,  1.0F,  9.0F,  9.0F,  20.0F, 20.0F,
                                        30.0F, 30.0F, 40.0F, 40.0F, 50.0F, 50.0F, 60.0F, 60.0F, 70.0F, 70.0F};
        std::vector<float> scoresData = {0.95F, 0.90F, 0.75F, 0.60F, 0.40F};
        std::vector<int32_t> indicesData = {0, 1, 2, 3, 4};
        std::vector<int32_t> maxOutputSizeData = {3};
        std::vector<float> iouThresholdData = {0.5F};
        std::vector<float> scoreThresholdData = {0.5F};
        std::vector<ge::Tensor> inputs = {MakeTensor(boxesDesc, boxesData),
                                          MakeTensor(scoresDesc, scoresData),
                                          MakeTensor(indicesDesc, indicesData),
                                          MakeTensor(scalarIntDesc, maxOutputSizeData),
                                          MakeTensor(scalarFloatDesc, iouThresholdData),
                                          MakeTensor(scalarFloatDesc, scoreThresholdData)};

        const std::map<ge::AscendString, ge::AscendString> sessionOptions;
        const std::map<ge::AscendString, ge::AscendString> graphOptions;
        std::unique_ptr<ge::Session> session(new (std::nothrow) ge::Session(sessionOptions));
        if (session == nullptr) {
            std::cerr << "Failed to create GE session" << std::endl;
        } else if ((ret = session->AddGraph(0, graph, graphOptions)) != ge::GRAPH_SUCCESS) {
            std::cerr << "AddGraph failed: " << ret << std::endl;
        } else {
            std::vector<ge::Tensor> outputs;
            ret = session->RunGraph(0, inputs, outputs);
            if (ret != ge::GRAPH_SUCCESS) {
                std::cerr << "RunGraph failed: " << ret << std::endl;
            } else if (CheckOutput(outputs)) {
                std::cout << "SortedNMS GEIR example passed" << std::endl;
                result = kSuccess;
            }
        }
    }

    ret = ge::GEFinalize();
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "GEFinalize failed: " << ret << std::endl;
        return kFailed;
    }
    return result;
}
