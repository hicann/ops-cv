/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2_backward.h"

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

namespace {
constexpr float kCompareTolerance = 1e-5F;
constexpr size_t kRoiElemNum = 5U;

struct AxisPoint {
    int64_t low = 0;
    int64_t high = 0;
    float lowWeight = 0.0F;
    float highWeight = 0.0F;
};

struct TestCase {
    std::string name;
    std::vector<int64_t> gradOutputShape;
    std::vector<float> gradOutputData;
    std::vector<int64_t> boxesShape;
    std::vector<float> boxesData;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> gradInputShape;
    int64_t pooledHeight = 0;
    int64_t pooledWidth = 0;
    float spatialScale = 0.0F;
    int64_t samplingRatio = 0;
    bool aligned = false;
    bool exactCheck = false;
    bool useCpuReference = false;
    bool expectWorkspaceSuccess = true;
    std::vector<float> expectedData;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int64_t OffsetNchw(const std::vector<int64_t>& shape, int64_t n, int64_t c, int64_t h, int64_t w)
{
    return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

const std::vector<int64_t>& GetGradInputShape(const TestCase& testCase)
{
    return testCase.gradInputShape.empty() ? testCase.inputShape : testCase.gradInputShape;
}

int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclFormat format,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    const auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INTERNAL_ERROR);
    return ACL_SUCCESS;
}

void DestroyCaseResource(aclTensor* tensor, void* deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

std::vector<float> MakeLinearData(size_t count, float start, float step)
{
    std::vector<float> data(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        data[i] = start + static_cast<float>(i) * step;
    }
    return data;
}

AxisPoint CalcAxisPoint(float coord, int64_t limit, int64_t samples)
{
    AxisPoint point;
    if (limit <= 0 || samples <= 0) {
        return point;
    }
    if (coord < -1.0F || coord > static_cast<float>(limit)) {
        return point;
    }

    const float clampedCoord = std::min(std::max(coord, 0.0F), static_cast<float>(limit - 1));
    const int64_t low = static_cast<int64_t>(std::floor(clampedCoord));
    const int64_t high = std::min(low + 1, limit - 1);
    const float frac = clampedCoord - static_cast<float>(low);

    point.low = low;
    point.high = high;
    point.lowWeight = (1.0F - frac) / static_cast<float>(samples);
    point.highWeight = frac / static_cast<float>(samples);
    return point;
}

std::vector<float> ComputeReference(const TestCase& testCase)
{
    std::vector<float> gradInput(static_cast<size_t>(GetShapeSize(testCase.inputShape)), 0.0F);
    const int64_t roiCount = testCase.boxesShape[0];
    const int64_t channels = testCase.inputShape[1];
    const int64_t inputH = testCase.inputShape[2];
    const int64_t inputW = testCase.inputShape[3];

    for (int64_t roiIndex = 0; roiIndex < roiCount; ++roiIndex) {
        const size_t roiOffset = static_cast<size_t>(roiIndex) * kRoiElemNum;
        const int64_t fmIndex = static_cast<int64_t>(std::floor(testCase.boxesData[roiOffset]));
        if (fmIndex < 0 || fmIndex >= testCase.inputShape[0]) {
            continue;
        }

        float roiStartX = testCase.boxesData[roiOffset + 1] * testCase.spatialScale;
        float roiStartY = testCase.boxesData[roiOffset + 2] * testCase.spatialScale;
        float roiEndX = testCase.boxesData[roiOffset + 3] * testCase.spatialScale;
        float roiEndY = testCase.boxesData[roiOffset + 4] * testCase.spatialScale;

        if (testCase.aligned) {
            roiStartX -= 0.5F;
            roiStartY -= 0.5F;
            roiEndX -= 0.5F;
            roiEndY -= 0.5F;
        }

        float roiWidth = roiEndX - roiStartX;
        float roiHeight = roiEndY - roiStartY;
        if (!testCase.aligned) {
            roiWidth = std::max(roiWidth, 1.0F);
            roiHeight = std::max(roiHeight, 1.0F);
        }

        const float binSizeW = roiWidth / static_cast<float>(testCase.pooledWidth);
        const float binSizeH = roiHeight / static_cast<float>(testCase.pooledHeight);
        const int64_t sampleW = testCase.samplingRatio > 0 ? testCase.samplingRatio :
                                                             static_cast<int64_t>(std::ceil(binSizeW));
        const int64_t sampleH = testCase.samplingRatio > 0 ? testCase.samplingRatio :
                                                             static_cast<int64_t>(std::ceil(binSizeH));
        if (sampleW <= 0 || sampleH <= 0) {
            continue;
        }

        const float gridW = binSizeW / static_cast<float>(sampleW);
        const float gridH = binSizeH / static_cast<float>(sampleH);

        for (int64_t poolH = 0; poolH < testCase.pooledHeight; ++poolH) {
            for (int64_t gridHIdx = 0; gridHIdx < sampleH; ++gridHIdx) {
                const float y = roiStartY + static_cast<float>(poolH) * binSizeH +
                                (static_cast<float>(gridHIdx) + 0.5F) * gridH;
                const AxisPoint yPoint = CalcAxisPoint(y, inputH, sampleH);
                if (yPoint.lowWeight == 0.0F && yPoint.highWeight == 0.0F) {
                    continue;
                }

                for (int64_t poolW = 0; poolW < testCase.pooledWidth; ++poolW) {
                    for (int64_t gridWIdx = 0; gridWIdx < sampleW; ++gridWIdx) {
                        const float x = roiStartX + static_cast<float>(poolW) * binSizeW +
                                        (static_cast<float>(gridWIdx) + 0.5F) * gridW;
                        const AxisPoint xPoint = CalcAxisPoint(x, inputW, sampleW);
                        if (xPoint.lowWeight == 0.0F && xPoint.highWeight == 0.0F) {
                            continue;
                        }

                        const float w1 = yPoint.lowWeight * xPoint.lowWeight;
                        const float w2 = yPoint.lowWeight * xPoint.highWeight;
                        const float w3 = yPoint.highWeight * xPoint.lowWeight;
                        const float w4 = yPoint.highWeight * xPoint.highWeight;

                        for (int64_t channel = 0; channel < channels; ++channel) {
                            const int64_t gradOffset = OffsetNchw(testCase.gradOutputShape, roiIndex, channel, poolH,
                                                                  poolW);
                            const float gradValue = testCase.gradOutputData[static_cast<size_t>(gradOffset)];
                            gradInput[static_cast<size_t>(OffsetNchw(testCase.inputShape, fmIndex, channel, yPoint.low,
                                                                     xPoint.low))] += gradValue * w1;
                            gradInput[static_cast<size_t>(OffsetNchw(testCase.inputShape, fmIndex, channel, yPoint.low,
                                                                     xPoint.high))] += gradValue * w2;
                            gradInput[static_cast<size_t>(OffsetNchw(testCase.inputShape, fmIndex, channel, yPoint.high,
                                                                     xPoint.low))] += gradValue * w3;
                            gradInput[static_cast<size_t>(OffsetNchw(testCase.inputShape, fmIndex, channel, yPoint.high,
                                                                     xPoint.high))] += gradValue * w4;
                        }
                    }
                }
            }
        }
    }

    return gradInput;
}

bool CompareResult(const std::string& caseName, const std::vector<float>& resultData,
                   const std::vector<float>& expectedData)
{
    if (resultData.size() != expectedData.size()) {
        LOG_PRINT("[%s] result size mismatch, expect %zu but got %zu\n", caseName.c_str(), expectedData.size(),
                  resultData.size());
        return false;
    }

    bool resultMatch = true;
    float maxAbsDiff = 0.0F;
    size_t maxDiffIndex = 0U;
    size_t mismatchCount = 0U;
    for (size_t i = 0; i < resultData.size(); ++i) {
        const float absDiff = std::fabs(resultData[i] - expectedData[i]);
        if (absDiff > maxAbsDiff) {
            maxAbsDiff = absDiff;
            maxDiffIndex = i;
        }
        if (absDiff > kCompareTolerance) {
            resultMatch = false;
            if (mismatchCount < 8U) {
                LOG_PRINT("[%s] mismatch at index %zu, expect %f but got %f\n", caseName.c_str(), i, expectedData[i],
                          resultData[i]);
            }
            ++mismatchCount;
        }
    }

    const size_t previewCount = std::min<size_t>(8U, resultData.size());
    for (size_t i = 0; i < previewCount; ++i) {
        LOG_PRINT("[%s] preview result[%zu] = %f\n", caseName.c_str(), i, resultData[i]);
    }

    if (resultMatch) {
        LOG_PRINT("[%s] result check passed.\n", caseName.c_str());
    } else {
        LOG_PRINT("[%s] result check failed, mismatch_count=%zu, max_abs_diff=%f at index %zu.\n", caseName.c_str(),
                  mismatchCount, maxAbsDiff, maxDiffIndex);
    }
    return resultMatch;
}

bool ValidateSmokeResult(const std::string& caseName, const std::vector<float>& resultData)
{
    if (resultData.empty()) {
        LOG_PRINT("[%s] smoke check failed, result is empty.\n", caseName.c_str());
        return false;
    }

    bool hasNonZero = false;
    const size_t previewCount = std::min<size_t>(8U, resultData.size());
    for (size_t i = 0; i < resultData.size(); ++i) {
        const float value = resultData[i];
        if (!std::isfinite(value)) {
            LOG_PRINT("[%s] smoke check failed, result[%zu] is not finite: %f\n", caseName.c_str(), i, value);
            return false;
        }
        if (std::fabs(value) > kCompareTolerance) {
            hasNonZero = true;
        }
        if (i < previewCount) {
            LOG_PRINT("[%s] smoke preview result[%zu] = %f\n", caseName.c_str(), i, value);
        }
    }

    if (!hasNonZero) {
        LOG_PRINT("[%s] smoke check failed, all result values are zero.\n", caseName.c_str());
        return false;
    }

    LOG_PRINT("[%s] smoke check passed.\n", caseName.c_str());
    return true;
}

bool RunSingleCase(const TestCase& testCase, aclrtStream stream)
{
    LOG_PRINT("========== Run case: %s ==========\n", testCase.name.c_str());

    bool success = false;
    void* gradOutputDeviceAddr = nullptr;
    void* boxesDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* boxes = nullptr;
    aclTensor* gradInput = nullptr;
    const aclIntArray* inputShapeArray = nullptr;
    const auto& gradInputShape = GetGradInputShape(testCase);
    std::vector<float> gradInputInitData(static_cast<size_t>(GetShapeSize(gradInputShape)), 0.0F);
    std::vector<float> resultData;

    do {
        auto ret = CreateAclTensor(testCase.gradOutputData, testCase.gradOutputShape, ACL_FORMAT_NCHW,
                                   &gradOutputDeviceAddr, ACL_FLOAT, &gradOutput);
        if (ret != ACL_SUCCESS) {
            break;
        }

        ret = CreateAclTensor(testCase.boxesData, testCase.boxesShape, ACL_FORMAT_ND, &boxesDeviceAddr, ACL_FLOAT,
                              &boxes);
        if (ret != ACL_SUCCESS) {
            break;
        }

        ret = CreateAclTensor(gradInputInitData, gradInputShape, ACL_FORMAT_NCHW, &gradInputDeviceAddr, ACL_FLOAT,
                              &gradInput);
        if (ret != ACL_SUCCESS) {
            break;
        }

        inputShapeArray = aclCreateIntArray(testCase.inputShape.data(), testCase.inputShape.size());
        if (inputShapeArray == nullptr) {
            LOG_PRINT("[%s] aclCreateIntArray failed.\n", testCase.name.c_str());
            break;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnRoiAlignV2BackwardGetWorkspaceSize(
            gradOutput, boxes, inputShapeArray, testCase.pooledHeight, testCase.pooledWidth, testCase.spatialScale,
            testCase.samplingRatio, testCase.aligned, gradInput, &workspaceSize, &executor);
        if (!testCase.expectWorkspaceSuccess) {
            if (ret == ACL_SUCCESS) {
                LOG_PRINT("[%s] expected GetWorkspaceSize to fail, but it succeeded.\n", testCase.name.c_str());
                break;
            }
            LOG_PRINT("[%s] expected failure observed, GetWorkspaceSize ret=%d.\n", testCase.name.c_str(), ret);
            success = true;
            break;
        }
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[%s] aclnnRoiAlignV2BackwardGetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
            break;
        }

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("[%s] allocate workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
                break;
            }
        }

        ret = aclnnRoiAlignV2Backward(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[%s] aclnnRoiAlignV2Backward failed. ERROR: %d\n", testCase.name.c_str(), ret);
            break;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[%s] aclrtSynchronizeStream failed. ERROR: %d\n", testCase.name.c_str(), ret);
            break;
        }

        resultData.assign(static_cast<size_t>(GetShapeSize(gradInputShape)), 0.0F);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                          resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[%s] copy resultData from device to host failed. ERROR: %d\n", testCase.name.c_str(), ret);
            break;
        }

        if (!testCase.exactCheck) {
            success = ValidateSmokeResult(testCase.name, resultData);
        } else if (testCase.useCpuReference) {
            success = CompareResult(testCase.name, resultData, ComputeReference(testCase));
        } else {
            success = CompareResult(testCase.name, resultData, testCase.expectedData);
        }
    } while (false);

    DestroyCaseResource(gradOutput, gradOutputDeviceAddr);
    DestroyCaseResource(boxes, boxesDeviceAddr);
    DestroyCaseResource(gradInput, gradInputDeviceAddr);
    if (inputShapeArray != nullptr) {
        aclDestroyIntArray(inputShapeArray);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return success;
}

std::vector<TestCase> BuildTestCases()
{
    std::vector<TestCase> cases;

    TestCase baseCase;
    baseCase.name = "base_aligned_false";
    baseCase.gradOutputShape = {1, 1, 3, 3};
    baseCase.gradOutputData = {4.5F, 6.5F, 8.5F, 16.5F, 18.5F, 20.5F, 28.5F, 30.5F, 32.5F};
    baseCase.boxesShape = {1, 5};
    baseCase.boxesData = {0.0F, -2.0F, -2.0F, 22.0F, 22.0F};
    baseCase.inputShape = {1, 1, 6, 6};
    baseCase.pooledHeight = 3;
    baseCase.pooledWidth = 3;
    baseCase.spatialScale = 0.25F;
    baseCase.samplingRatio = 2;
    baseCase.aligned = false;
    baseCase.exactCheck = true;
    baseCase.useCpuReference = false;
    baseCase.expectedData = {1.125F, 1.125F, 1.625F, 1.625F, 2.125F, 2.125F, 1.125F, 1.125F, 1.625F,
                             1.625F, 2.125F, 2.125F, 4.125F, 4.125F, 4.625F, 4.625F, 5.125F, 5.125F,
                             4.125F, 4.125F, 4.625F, 4.625F, 5.125F, 5.125F, 7.125F, 7.125F, 7.625F,
                             7.625F, 8.125F, 8.125F, 7.125F, 7.125F, 7.625F, 7.625F, 8.125F, 8.125F};
    cases.push_back(baseCase);

    TestCase alignedCase = baseCase;
    alignedCase.name = "aligned_true";
    alignedCase.boxesData = {0.0F, 0.0F, 0.0F, 16.0F, 16.0F};
    alignedCase.aligned = true;
    alignedCase.useCpuReference = true;
    alignedCase.expectedData.clear();
    cases.push_back(alignedCase);

    TestCase dynamicCase;
    dynamicCase.name = "dynamic_sampling_multi_roi_batch";
    dynamicCase.gradOutputShape = {3, 2, 2, 2};
    dynamicCase.gradOutputData = MakeLinearData(static_cast<size_t>(GetShapeSize(dynamicCase.gradOutputShape)), 1.0F,
                                                0.25F);
    dynamicCase.boxesShape = {3, 5};
    dynamicCase.boxesData = {0.0F,  0.0F,  0.0F, 6.0F,  6.0F, 1.0F, 2.0F, 2.0F,
                             10.0F, 10.0F, 0.0F, -2.0F, 1.0F, 8.0F, 11.0F};
    dynamicCase.inputShape = {2, 2, 6, 6};
    dynamicCase.pooledHeight = 2;
    dynamicCase.pooledWidth = 2;
    dynamicCase.spatialScale = 0.5F;
    dynamicCase.samplingRatio = 0;
    dynamicCase.aligned = false;
    dynamicCase.exactCheck = true;
    dynamicCase.useCpuReference = true;
    cases.push_back(dynamicCase);

    TestCase multiC1Case;
    multiC1Case.name = "multi_roi_multi_c1";
    multiC1Case.gradOutputShape = {2, 32, 2, 2};
    multiC1Case.gradOutputData = MakeLinearData(static_cast<size_t>(GetShapeSize(multiC1Case.gradOutputShape)), 0.1F,
                                                0.05F);
    multiC1Case.boxesShape = {2, 5};
    multiC1Case.boxesData = {0.0F, 0.0F, 0.0F, 12.0F, 12.0F, 0.0F, 2.0F, 2.0F, 14.0F, 14.0F};
    multiC1Case.inputShape = {1, 32, 4, 4};
    multiC1Case.pooledHeight = 2;
    multiC1Case.pooledWidth = 2;
    multiC1Case.spatialScale = 0.25F;
    multiC1Case.samplingRatio = 2;
    multiC1Case.aligned = false;
    multiC1Case.exactCheck = true;
    multiC1Case.useCpuReference = true;
    cases.push_back(multiC1Case);

    TestCase invalidSampleNumCase = baseCase;
    invalidSampleNumCase.name = "invalid_sampling_ratio_neg2";
    invalidSampleNumCase.samplingRatio = -2;
    invalidSampleNumCase.exactCheck = false;
    invalidSampleNumCase.expectWorkspaceSuccess = false;
    invalidSampleNumCase.expectedData.clear();
    cases.push_back(invalidSampleNumCase);

    TestCase invalidSpatialScaleCase = baseCase;
    invalidSpatialScaleCase.name = "invalid_spatial_scale_zero";
    invalidSpatialScaleCase.spatialScale = 0.0F;
    invalidSpatialScaleCase.exactCheck = false;
    invalidSpatialScaleCase.expectWorkspaceSuccess = false;
    invalidSpatialScaleCase.expectedData.clear();
    cases.push_back(invalidSpatialScaleCase);

    TestCase invalidPooledHeightCase = baseCase;
    invalidPooledHeightCase.name = "invalid_pooled_height_zero";
    invalidPooledHeightCase.pooledHeight = 0;
    invalidPooledHeightCase.exactCheck = false;
    invalidPooledHeightCase.expectWorkspaceSuccess = false;
    invalidPooledHeightCase.expectedData.clear();
    cases.push_back(invalidPooledHeightCase);

    TestCase invalidPooledWidthCase = baseCase;
    invalidPooledWidthCase.name = "invalid_pooled_width_zero";
    invalidPooledWidthCase.pooledWidth = 0;
    invalidPooledWidthCase.exactCheck = false;
    invalidPooledWidthCase.expectWorkspaceSuccess = false;
    invalidPooledWidthCase.expectedData.clear();
    cases.push_back(invalidPooledWidthCase);

    TestCase invalidBoxesShapeCase = baseCase;
    invalidBoxesShapeCase.name = "invalid_boxes_cols_4";
    invalidBoxesShapeCase.boxesShape = {1, 4};
    invalidBoxesShapeCase.boxesData = {0.0F, -2.0F, -2.0F, 22.0F};
    invalidBoxesShapeCase.exactCheck = false;
    invalidBoxesShapeCase.expectWorkspaceSuccess = false;
    invalidBoxesShapeCase.expectedData.clear();
    cases.push_back(invalidBoxesShapeCase);

    TestCase invalidRoiCountCase = baseCase;
    invalidRoiCountCase.name = "invalid_roi_count_mismatch";
    invalidRoiCountCase.gradOutputShape = {2, 1, 3, 3};
    invalidRoiCountCase.gradOutputData = MakeLinearData(
        static_cast<size_t>(GetShapeSize(invalidRoiCountCase.gradOutputShape)), 1.0F, 0.5F);
    invalidRoiCountCase.exactCheck = false;
    invalidRoiCountCase.expectWorkspaceSuccess = false;
    invalidRoiCountCase.expectedData.clear();
    cases.push_back(invalidRoiCountCase);

    TestCase invalidYDiffChannelCase = baseCase;
    invalidYDiffChannelCase.name = "invalid_y_diff_c_mismatch";
    invalidYDiffChannelCase.gradOutputShape = {1, 2, 3, 3};
    invalidYDiffChannelCase.gradOutputData = MakeLinearData(
        static_cast<size_t>(GetShapeSize(invalidYDiffChannelCase.gradOutputShape)), 1.0F, 0.5F);
    invalidYDiffChannelCase.exactCheck = false;
    invalidYDiffChannelCase.expectWorkspaceSuccess = false;
    invalidYDiffChannelCase.expectedData.clear();
    cases.push_back(invalidYDiffChannelCase);

    TestCase invalidYDiffHwCase = baseCase;
    invalidYDiffHwCase.name = "invalid_y_diff_hw_mismatch";
    invalidYDiffHwCase.gradOutputShape = {1, 1, 2, 3};
    invalidYDiffHwCase.gradOutputData = MakeLinearData(
        static_cast<size_t>(GetShapeSize(invalidYDiffHwCase.gradOutputShape)), 1.0F, 0.5F);
    invalidYDiffHwCase.exactCheck = false;
    invalidYDiffHwCase.expectWorkspaceSuccess = false;
    invalidYDiffHwCase.expectedData.clear();
    cases.push_back(invalidYDiffHwCase);

    TestCase invalidInputShapeRankCase = baseCase;
    invalidInputShapeRankCase.name = "invalid_input_shape_rank_3";
    invalidInputShapeRankCase.inputShape = {1, 6, 6};
    invalidInputShapeRankCase.gradInputShape = baseCase.inputShape;
    invalidInputShapeRankCase.exactCheck = false;
    invalidInputShapeRankCase.expectWorkspaceSuccess = false;
    invalidInputShapeRankCase.expectedData.clear();
    cases.push_back(invalidInputShapeRankCase);

    TestCase invalidGradInputShapeCase = baseCase;
    invalidGradInputShapeCase.name = "invalid_grad_input_shape_mismatch";
    invalidGradInputShapeCase.gradInputShape = {1, 1, 6, 5};
    invalidGradInputShapeCase.exactCheck = false;
    invalidGradInputShapeCase.expectWorkspaceSuccess = false;
    invalidGradInputShapeCase.expectedData.clear();
    cases.push_back(invalidGradInputShapeCase);

    return cases;
}
} // namespace

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    bool allPass = true;
    for (const auto& testCase : BuildTestCases()) {
        if (!RunSingleCase(testCase, stream)) {
            allPass = false;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (!allPass) {
        LOG_PRINT("roi_align_grad example failed.\n");
        return 1;
    }
    LOG_PRINT("roi_align_grad example passed.\n");
    return 0;
}
