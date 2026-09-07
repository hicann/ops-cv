/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_3d.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int32_t WARMUP = 5;
constexpr int32_t REPEAT = 20;

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const auto dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
aclTensor* CreateTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dtype,
                        void** deviceAddr)
{
    const size_t bytes = hostData.size() * sizeof(T);
    if (aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_NCDHW, shape.data(),
                           shape.size(), *deviceAddr);
}

template <typename T>
bool CheckNearest2x(const std::vector<T>& input, const std::vector<T>& output)
{
    constexpr int64_t channels = 3;
    constexpr int64_t inputD = 64;
    constexpr int64_t inputH = 64;
    constexpr int64_t inputW = 64;
    constexpr int64_t outputD = 128;
    constexpr int64_t outputH = 128;
    constexpr int64_t outputW = 128;
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t d = 0; d < outputD; ++d) {
            for (int64_t h = 0; h < outputH; ++h) {
                for (int64_t w = 0; w < outputW; ++w) {
                    const int64_t inputIndex = ((c * inputD + d / 2) * inputH + h / 2) * inputW + w / 2;
                    const int64_t outputIndex = ((c * outputD + d) * outputH + h) * outputW + w;
                    if (input[inputIndex] != output[outputIndex]) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <typename T>
double RunCase(const char* name, aclDataType dtype, aclrtStream stream, bool& exact)
{
    const std::vector<int64_t> inputShape = {1, 3, 64, 64, 64};
    const std::vector<int64_t> outputShape = {1, 3, 128, 128, 128};
    const std::vector<int64_t> outputSizeData = {128, 128, 128};
    std::vector<T> input(ShapeSize(inputShape));
    std::vector<T> output(ShapeSize(outputShape), 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<T>(i % 251);
    }
    input[0] = static_cast<T>(0);
    input[1] = static_cast<T>(255);

    void* inputAddr = nullptr;
    void* outputAddr = nullptr;
    aclTensor* inputTensor = CreateTensor(input, inputShape, dtype, &inputAddr);
    aclTensor* outputTensor = CreateTensor(output, outputShape, dtype, &outputAddr);
    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    if (inputTensor == nullptr || outputTensor == nullptr || outputSize == nullptr) {
        return -1.0;
    }

    void* workspace = nullptr;
    uint64_t workspaceCapacity = 0;
    auto invoke = [&]() -> bool {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnUpsampleNearest3dGetWorkspaceSize(inputTensor, outputSize, 0.0, 0.0, 0.0, outputTensor, &workspaceSize,
                                                   &executor) != ACL_SUCCESS) {
            return false;
        }
        if (workspaceSize > workspaceCapacity) {
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                return false;
            }
            workspaceCapacity = workspaceSize;
        }
        return aclnnUpsampleNearest3d(workspace, workspaceSize, executor, stream) == ACL_SUCCESS &&
               aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    };

    for (int32_t i = 0; i < WARMUP; ++i) {
        if (!invoke()) {
            return -1.0;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < REPEAT; ++i) {
        if (!invoke()) {
            return -1.0;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double averageUs = std::chrono::duration<double, std::micro>(end - start).count() / REPEAT;

    aclrtMemcpy(output.data(), output.size() * sizeof(T), outputAddr, output.size() * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST);
    exact = CheckNearest2x(input, output);
    std::cout << name << " exact_match=" << (exact ? "YES" : "NO") << " average_us=" << averageUs << '\n';

    aclDestroyIntArray(outputSize);
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclrtFree(inputAddr);
    aclrtFree(outputAddr);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return averageUs;
}
} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(DEVICE_ID) != ACL_SUCCESS) {
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return 1;
    }
    std::cout << "Device: Atlas A2 / Ascend 910B3\n";
    std::cout << "Case: input=[1,3,64,64,64], output=[1,3,128,128,128], warmup=" << WARMUP << ", repeat=" << REPEAT
              << '\n';
    bool uint8Exact = false;
    bool fp16Exact = false;
    const double uint8Us = RunCase<uint8_t>("UINT8", ACL_UINT8, stream, uint8Exact);
    const double fp16Us = RunCase<aclFloat16>("FP16", ACL_FLOAT16, stream, fp16Exact);
    const double degradation = (uint8Us / fp16Us - 1.0) * 100.0;
    std::cout << "UINT8_vs_FP16_degradation=" << degradation << "%\n";
    const bool passed = uint8Exact && fp16Exact && uint8Us > 0.0 && fp16Us > 0.0 && degradation <= 5.0;
    std::cout << "PERFORMANCE_THRESHOLD(<=5%): " << (degradation <= 5.0 ? "PASS" : "FAIL") << '\n';
    std::cout << "RESULT: " << (passed ? "PASS" : "FAIL") << '\n';
    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return passed ? 0 : 1;
}
