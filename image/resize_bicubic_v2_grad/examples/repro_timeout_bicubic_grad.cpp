/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repro_timeout_bicubic_grad.cpp
 * \brief On-board reproduction for the ResizeBicubicV2Grad vector core timeout (issue-1).
 *
 * Bug shape: grads (gradOut) (1,1,2147483649,1) -> y (gradInput) (1,1,2,1), NCHW, fp32.
 *   outputSize = spatial of gradOut = [2147483649, 1]; inputSize = full shape of gradInput = [1,1,2,1].
 * Before the split-K fix the SimtDetermine path assigns only yShapeSize(=2) active threads, each
 * serially scanning the ~2^31 H gather domain, tripping the vector core watchdog. With the fix the
 * tiling selects the deterministic split-K path (tilingKey 20003) and the reduction is parallelized
 * across cores/threads, so the stream synchronizes without timeout.
 *
 * Allocates the 8.6GB grads directly on device (memset) to avoid a huge host vector.
 * Run after the fixed ResizeBicubicV2Grad kernel package is installed as the active vendor op.
 */
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bicubic_2d_backward.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

#define LOG(fmt, ...)               \
    do {                            \
        printf(fmt, ##__VA_ARGS__); \
        fflush(stdout);             \
    } while (0)

static int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto v : s)
        n *= v;
    return n;
}

static aclTensor* MakeDeviceTensor(const std::vector<int64_t>& shape, aclDataType dtype, void** devAddr)
{
    int64_t elemCnt = ShapeSize(shape);
    size_t bytes = static_cast<size_t>(elemCnt) * sizeof(float);
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMalloc failed (%zu bytes). ERROR: %d\n", bytes, ret); return nullptr);
    ret = aclrtMemset(*devAddr, bytes, 0, bytes);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMemset failed. ERROR: %d\n", ret); return nullptr);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                           shape.data(), shape.size(), *devAddr);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    const int64_t bigH = 2147483649LL; // INT32_MAX + 2

    std::vector<int64_t> gradOutShape = {1, 1, bigH, 1}; // self / grads
    std::vector<int64_t> gradInShape = {1, 1, 2, 1};     // out / y

    void* gradOutDev = nullptr;
    void* gradInDev = nullptr;
    aclTensor* gradOut = MakeDeviceTensor(gradOutShape, aclDataType::ACL_FLOAT, &gradOutDev);
    CHECK_RET(gradOut != nullptr, return -1);
    aclTensor* gradIn = MakeDeviceTensor(gradInShape, aclDataType::ACL_FLOAT, &gradInDev);
    CHECK_RET(gradIn != nullptr, return -1);

    std::vector<int64_t> outputSizeArr = {bigH, 1}; // spatial of gradOut
    const aclIntArray* outputSize = aclCreateIntArray(outputSizeArr.data(), outputSizeArr.size());
    std::vector<int64_t> inputSizeArr = {1, 1, 2, 1}; // full shape of gradInput
    const aclIntArray* inputSize = aclCreateIntArray(inputSizeArr.data(), inputSizeArr.size());
    CHECK_RET(outputSize != nullptr && inputSize != nullptr, return -1);

    bool alignCorners = false;
    double scalesH = 0.0;
    double scalesW = 0.0;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    LOG("[repro] calling GetWorkspaceSize...\n");
    ret = aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW,
                                                         gradIn, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG("[repro] workspaceSize=%lu\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG("workspace malloc failed. ERROR: %d\n", ret); return ret);
    }

    LOG("[repro] launching kernel...\n");
    auto t0 = std::chrono::steady_clock::now();
    ret = aclnnUpsampleBicubic2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclnnUpsampleBicubic2dBackward failed. ERROR: %d\n", ret); return ret);

    LOG("[repro] synchronizing stream (pre-fix: watchdog timeout here)...\n");
    ret = aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    LOG("[repro] sync returned ret=%d after %.3f s\n", ret, secs);
    CHECK_RET(ret == ACL_SUCCESS, LOG("SynchronizeStream failed (TIMEOUT?). ERROR: %d\n", ret); return ret);

    LOG("[repro] completed without timeout.\n");

    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradIn);
    aclDestroyIntArray(outputSize);
    aclDestroyIntArray(inputSize);
    aclrtFree(gradOutDev);
    aclrtFree(gradInDev);
    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
