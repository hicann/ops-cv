/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_iou.h"

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
  // 固定写法，AscendCL初始化
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> bBoxesHostData = {1.0, 1.0, 5.0, 3.0, 1.0, 1.0, 5.0, 3.0};
  std::vector<float> gtBoxesHostData = {4.0, 2.0, 9.0, 5.0, 4.0, 2.0, 9.0, 5.0};
  std::vector<float> overlapHostData = {0.045455, 0.045455};
  std::vector<int64_t> bBoxesShape = {2, 4};
  std::vector<int64_t> gtBoxesShape = {2, 4};
  std::vector<int64_t> overlapShape = {2, 1};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  const char* mode = "iou";
  float eps = 0.0f;
  bool aligned = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // ret = aclnnIouGetWorkspaceSize(bBoxes, gtBoxes, mode, eps, aligned, overlap, &workspaceSize, &executor);
  // CHECK_RET(
  //     ret == ACL_SUCCESS,
  //     LOG_PRINT("aclnnIouGetWorkspaceSize failed. ERROR: %d\n", ret);
  //     return ret);

  // // 根据第一段接口计算出的workspaceSize申请device内存
  // void* workspaceAddr = nullptr;
  // if (workspaceSize > 0) {
  //   ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  //   CHECK_RET(ret == ACL_SUCCESS,
  //             LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
  //             return ret);
  // }

  // // aclnnIou
  // ret = aclnnIou(workspaceAddr, workspaceSize, executor, stream);
  // CHECK_RET(ret == ACL_SUCCESS,
  //           LOG_PRINT("aclnnIou failed. ERROR: %d\n", ret);
  //           return ret);

  // // 4. （固定写法）同步等待任务执行结束
  // ret = aclrtSynchronizeStream(stream);
  // CHECK_RET(ret == ACL_SUCCESS,
  //           LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
  //           return ret);

  // // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // auto size = GetShapeSize(overlapShape);
  // std::vector<float> resultData(size, 0);
  // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), overlapDeviceAddr,
  //                   size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  // CHECK_RET(ret == ACL_SUCCESS,
  //           LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
  //           return ret);
  // for (int64_t i = 0; i < size; i++) {
  //   LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  // }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);

  // 7. 释放device资源
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);

  // if (workspaceSize > 0) {
  //   aclrtFree(workspaceAddr);
  // }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}