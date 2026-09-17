/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_context_faker.h"

namespace {

struct InferResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    std::vector<int64_t> output;
};

InferResult RunInfer(int64_t pooledHeight, int64_t pooledWidth)
{
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(2, 1);
    std::vector<std::unique_ptr<gert::Tensor>> keepAlive;
    std::vector<gert::Tensor*> inputs;
    for (const gert::StorageShape& storageShape :
         {gert::StorageShape({1, 3, 4, 4}, {1, 3, 4, 4}), gert::StorageShape({1, 5}, {1, 5})}) {
        auto tensor = std::make_unique<gert::Tensor>(
            storageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
        inputs.push_back(tensor.get());
        keepAlive.push_back(std::move(tensor));
    }
    faker.InputTensors(inputs)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .Attr("pooled_height", pooledHeight)
        .Attr("pooled_width", pooledWidth)
        .Attr("spatial_scale", 1.0F)
        .Attr("sampling_ratio", static_cast<int64_t>(1))
        .SetOpType("RoiAlignV2");
    auto holder = faker.Build();

    InferResult result;
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto* impl = registry == nullptr ? nullptr : registry->GetOpImpl("RoiAlignV2");
    if (impl == nullptr || impl->infer_shape == nullptr) {
        return result;
    }
    result.status = impl->infer_shape(holder.GetContext());
    const auto* output = holder.GetContext()->GetOutputShape(0);
    if (output != nullptr) {
        for (size_t i = 0; i < output->GetDimNum(); ++i) {
            result.output.push_back(output->GetDim(i));
        }
    }
    return result;
}

void PrintCase(const char* name, int64_t pooledHeight, int64_t pooledWidth, const InferResult& result)
{
    std::cout << "case=" << name << " input=(" << pooledHeight << "," << pooledWidth
              << ") actual_status=" << static_cast<int64_t>(result.status) << " output=[";
    for (size_t i = 0; i < result.output.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << result.output[i];
    }
    std::cout << "] expected_contract=INT32 positive range only\n";
}

} // namespace

TEST(RoiAlignV2PooledSizeInt32Narrowing, accepts_int32_max_boundary)
{
    const int64_t maxInt32 = INT32_MAX;
    const auto result = RunInfer(maxInt32, 2);
    PrintCase("int32_max_boundary", maxInt32, 2, result);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.output, (std::vector<int64_t>{1, 3, maxInt32, 2}));
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, accepts_int32_max_width_boundary)
{
    const int64_t maxInt32 = INT32_MAX;
    const auto result = RunInfer(2, maxInt32);
    PrintCase("int32_max_width_boundary", 2, maxInt32, result);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.output, (std::vector<int64_t>{1, 3, 2, maxInt32}));
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, accepts_valid_control)
{
    const auto result = RunInfer(2, 2);
    PrintCase("valid_control", 2, 2, result);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.output, (std::vector<int64_t>{1, 3, 2, 2}));
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, rejects_positive_height_above_int32)
{
    const int64_t outOfRange = static_cast<int64_t>(INT32_MAX) + 1;
    const auto result = RunInfer(outOfRange, 2);
    PrintCase("positive_height_above_int32", outOfRange, 2, result);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    EXPECT_TRUE(result.output.empty());
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, rejects_positive_width_above_int32)
{
    const int64_t outOfRange = static_cast<int64_t>(INT32_MAX) + 1;
    const auto result = RunInfer(2, outOfRange);
    PrintCase("positive_width_above_int32", 2, outOfRange, result);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    EXPECT_TRUE(result.output.empty());
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, rejects_height_wraparound)
{
    const int64_t outOfRange = static_cast<int64_t>(UINT32_MAX) + 2;
    const auto result = RunInfer(outOfRange, 2);
    PrintCase("height_wraparound", outOfRange, 2, result);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    EXPECT_TRUE(result.output.empty());
}

TEST(RoiAlignV2PooledSizeInt32Narrowing, rejects_width_wraparound)
{
    const int64_t outOfRange = static_cast<int64_t>(UINT32_MAX) + 2;
    const auto result = RunInfer(2, outOfRange);
    PrintCase("width_wraparound", 2, outOfRange, result);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    EXPECT_TRUE(result.output.empty());
}
