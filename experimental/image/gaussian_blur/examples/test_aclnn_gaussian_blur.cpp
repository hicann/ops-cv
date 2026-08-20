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
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gaussian_blur.h"

#define CHECK_RET(condition, action) \
    do {                             \
        if (!(condition)) {          \
            action;                  \
        }                            \
    } while (0)

namespace {

int64_t GetElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

int CreateTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddress,
                 aclTensor** tensor)
{
    const size_t bytes = hostData.size() * sizeof(float);
    auto ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddress);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return 1);

    const std::vector<int64_t> shape = {64, 64, 3};
    const size_t elementCount = static_cast<size_t>(GetElementCount(shape));
    std::vector<float> input(elementCount);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 251U) / 251.0f;
    }
    std::vector<float> output(elementCount, 0.0f);

    void* srcAddress = nullptr;
    void* dstAddress = nullptr;
    void* workspaceAddress = nullptr;
    aclTensor* src = nullptr;
    aclTensor* dst = nullptr;
    aclIntArray* ksize = nullptr;

    CHECK_RET(CreateTensor(input, shape, &srcAddress, &src) == ACL_SUCCESS, return 1);
    CHECK_RET(CreateTensor(output, shape, &dstAddress, &dst) == ACL_SUCCESS, return 1);
    const int64_t kernelShape[] = {7, 7};
    ksize = aclCreateIntArray(kernelShape, 2);
    CHECK_RET(ksize != nullptr, return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnGaussianBlurGetWorkspaceSize(src, ksize, 1.5, 1.5, 4, dst, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("GetWorkspaceSize failed: %d\n", ret); return 1);
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspaceAddress, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 1);
    }

    ret = aclnnGaussianBlur(workspaceAddress, workspaceSize, executor, src, ksize, 1.5, 1.5, 4, dst, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnGaussianBlur failed: %d\n", ret); return 1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtMemcpy(output.data(), output.size() * sizeof(float), dstAddress, output.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return 1);

    std::printf("GaussianBlur succeeded, output[0]=%.6f\n", output[0]);
    aclDestroyIntArray(ksize);
    aclDestroyTensor(src);
    aclDestroyTensor(dst);
    aclrtFree(srcAddress);
    aclrtFree(dstAddress);
    if (workspaceAddress != nullptr) {
        aclrtFree(workspaceAddress);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
