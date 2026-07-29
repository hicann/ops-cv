/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UINT8 数据类型 aclnnUpsampleNearestExact2d 调用样例。
// 与 test_aclnn_upsample_nearest_exact2d.cpp 的区别仅在于：输入/输出数据类型为 ACL_UINT8，
// 主机数据使用 uint8_t。最近邻上采样对元素仅做拷贝搬运，uint8 结果应与同输入的浮点结果按位置一一对应。

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_exact2d.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造 uint8 输入与输出
    int64_t outSize = 32;
    std::vector<int64_t> selfShape = {1, 1, 4, 2};
    std::vector<int64_t> outShape = {1, 1, 8, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint8_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> outputSizeHostData = {8, 4};
    std::vector<uint8_t> outHostData(outSize, 0);

    double scalesH = 0.0;
    double scalesW = 0.0;

    // 创建 uint8 self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_UINT8, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto outputSizeArray = aclCreateIntArray(outputSizeHostData.data(), 2);
    // 创建 uint8 out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnUpsampleNearestExact2d
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnUpsampleNearestExact2dGetWorkspaceSize(self, outputSizeArray, scalesH, scalesW, out, &workspaceSize,
                                                      &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearestExact2dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnUpsampleNearestExact2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearestExact2d failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 拷回结果
    auto size = GetShapeSize(outShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, static_cast<uint32_t>(resultData[i]));
    }

    // 6. 释放资源
    aclDestroyTensor(self);
    aclDestroyIntArray(outputSizeArray);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
