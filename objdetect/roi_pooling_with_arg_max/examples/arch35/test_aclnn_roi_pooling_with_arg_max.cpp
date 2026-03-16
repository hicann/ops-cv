/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_with_arg_max.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorOutput(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor, aclFormat format) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {2, 2, 6, 8};
  std::vector<int64_t> roisShape = {2, 5};
  std::vector<int64_t> yShape = {2, 2, 2, 2};
  std::vector<int64_t> argmaxShape = {2, 2, 2, 2};

  void* xDeviceAddr = nullptr;
  void* roisDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* argmaxDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* rois = nullptr;
  aclTensor* y = nullptr;
  aclTensor* argmax = nullptr;

  int64_t xSize = GetShapeSize(xShape);
  int64_t roisSize = GetShapeSize(roisShape);
  std::vector<float> xHostData(xSize, 1.0f);
  std::vector<float> roisHostData(roisSize, 0.0f);
  roisHostData[0] = 0.0f;
  roisHostData[1] = 0.0f;
  roisHostData[2] = 0.0f;
  roisHostData[3] = 7.0f;
  roisHostData[4] = 5.0f;
  roisHostData[5] = 1.0f;
  roisHostData[6] = 0.0f;
  roisHostData[7] = 0.0f;
  roisHostData[8] = 7.0f;
  roisHostData[9] = 5.0f;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois,
                        aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<int32_t>(argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax,
                                         aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t pooledH = 2;
  int64_t pooledW = 2;
  float spatialScaleH = 1.0f;
  float spatialScaleW = 1.0f;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnRoiPoolingWithArgMaxGetWorkspaceSize(x, rois, pooledH, pooledW, spatialScaleH, spatialScaleW,
                                                  y, argmax, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnRoiPoolingWithArgMax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMax failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  int64_t yElem = GetShapeSize(yShape);
  std::vector<float> yResult(yElem, 0.0f);
  ret = aclrtMemcpy(yResult.data(), yElem * sizeof(float), yDeviceAddr, yElem * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

  int64_t argmaxElem = GetShapeSize(argmaxShape);
  std::vector<int32_t> argmaxResult(argmaxElem, 0);
  ret = aclrtMemcpy(argmaxResult.data(), argmaxElem * sizeof(int32_t), argmaxDeviceAddr,
                    argmaxElem * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy argmax from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("y: ");
  for (int64_t i = 0; i < yElem; i++) {
    LOG_PRINT("%f ", yResult[i]);
  }
  LOG_PRINT("\nargmax: ");
  for (int64_t i = 0; i < argmaxElem; i++) {
    LOG_PRINT("%d ", argmaxResult[i]);
  }
  LOG_PRINT("\n");

  aclDestroyTensor(x);
  aclDestroyTensor(rois);
  aclDestroyTensor(y);
  aclDestroyTensor(argmax);
  aclrtFree(xDeviceAddr);
  aclrtFree(roisDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(argmaxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
