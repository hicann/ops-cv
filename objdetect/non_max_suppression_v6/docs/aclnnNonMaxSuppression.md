# aclnnNonMaxSuppression

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/non_max_suppression_v6)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

算子功能：删除分数小于scoreThreshold的边界框，筛选出与之前被选中部分重叠较高（IOU较高）的框。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNonMaxSuppressionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNonMaxSuppression”接口执行计算。

- `aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, aclIntArray *maxOutputBoxesPerClass, aclFloatArray *iouThreshold, aclFloatArray *scoreThreshold, int32_t centerPointBox, aclTensor *selectedIndices, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnNonMaxSuppression(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNonMaxSuppressionGetWorkspaceSize

- **参数说明：**

  - boxes(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。shape为[num_batches, spatial_dimension, 4]。

  - scores(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。shape为[num_batches, num_classes, spatial_dimension]。

  - maxOutputBoxesPerClass(aclIntArray*, 计算输入)：表示每个批次每个类别选择的最大框数。Host侧的aclIntArray，数据类型支持INT32。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。数值上限为700。

  - iouThreshold(aclFloatArray*, 计算输入)：Host侧的aclFloatArray，数据类型支持FLOAT。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。取值范围[0, 1]。表示判断框相对于IOU是否重叠过多的阈值。

  - scoreThreshold(aclFloatArray*, 计算输入)：Host侧的aclFloatArray，数据类型支持FLOAT。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。取值范围[0, 1]。表示根据得分决定何时移除框的阈值。

  - centerPointBox(int, 计算输入)：Host侧的整数，数据类型支持INT32。取值范围[0, 1]。用于决定边界框格式。当等于0时，主要用于TensorFlow模型, 数据以(y1, x1, y2, x2)形式提供，其中(y1, x1) 、(y2, x2)是对角线框角坐标，需要用户自行保证x1<x2、y1<y2。当等于1时，主要用于PyTorch模型，数据以(x_center, y_center, width, height)形式提供。

  - selectedIndices(aclTensor*, 计算输出)：Device侧的aclTensor，数据类型支持INT32。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持[数据格式ND](../../../docs/zh/context/数据格式.md)。shape为[num_selected_indices, 3]，数据以[batch_index, class_index, box_index]形式提供。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回361001(ACLNN_ERR_RUNTIME_ERROR)：1. 当前产品不支持。
返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的boxes、scores、out是空指针。
返回161002(ACLNN_ERR_PARAM_INVALID)：1. boxes、scores和maxOutputBoxesPerClass的数据类型不在支持的范围内。
                                    2. boxes、scores和 selectedIndices 的数据格式不在支持的范围内。
                                    3. boxes、scores 的shape不在支持的范围内：
                                        1) boxes、scores需为3维
                                        2) boxes第0维必须等于scores第0维度
                                        3) boxes第1维必须等于scores第2维度
                                        4) boxes第2维必须等于4
                                    4. iouThreshold、scoreThreshold、centerPointBox、maxOutputBoxesPerClass数值不在支持的范围内。
```

## aclnnNonMaxSuppression

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnNonMaxSuppressionGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的Stream。


- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. maxOutputBoxesPerClass参数上限为700。输入参数boxes和scores的数据类型要求保持一致。
2. 在FLOAT16场景下，算子进行排序和计算对比标杆可能会引入计算误差。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_non_max_suppression.h"

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

template <typename T>
int64_t GetShapeSize(const std::vector<T>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 调用aclCreateIntArray接口创建aclIntArray
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> boxesShape = {1, 7, 4};
  std::vector<int64_t> scoresShape = {1, 1, 7};
  std::vector<int64_t> maxSizePerClassShape = {3};
  std::vector<int64_t> selectedIndicesShape = {3, 3};

  void* boxesDeviceAddr = nullptr;
  void* scoresDeviceAddr = nullptr;
  void* maxSizePerClassDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* boxes = nullptr;
  aclTensor* scores = nullptr;
  aclIntArray* maxOutputBoxesPerClass = nullptr;
  aclFloatArray* iouThd = nullptr;
  aclFloatArray* scoresThd = nullptr;
  aclTensor* selectedIndices = nullptr;

  std::vector<float> boxesHostData = {
    49.1, 32.4, 51.0, 35.9,
    49.3, 32.9, 51.0, 35.3,
    49.2, 31.8, 51.0, 35.4,
    35.1, 11.5, 39.1, 15.7, 
    35.6, 11.8, 39.3, 14.2,
    35.3, 11.5, 39.9, 14.5, 
    35.2, 11.7, 39.7, 15.7,
  };
  std::vector<float> scoresHostData = {0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3};
  std::vector<int64_t> maxOutputBoxesPerClassHostData = {3};
  std::vector<float> iouThresholdHostData = {0.6};
  std::vector<float> scoreThresholdHostData = {0};
  std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建aclTensor: boxes
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建aclTensor: scores
  ret = CreateAclTensor(scoresHostData, scoresShape, &scoresDeviceAddr, aclDataType::ACL_FLOAT, &scores);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建AclIntArray: maxOutputBoxesPerClass
  ret = CreateAclIntArray(maxOutputBoxesPerClassHostData, &maxSizePerClassDeviceAddr, &maxOutputBoxesPerClass);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建AclFloatArray: iouThreshold
  iouThd = aclCreateFloatArray(iouThresholdHostData.data(), iouThresholdHostData.size());
  CHECK_RET(iouThd != nullptr, return 0);

  // 创建AclFloatArray: scoresThreshold
  scoresThd = aclCreateFloatArray(scoreThresholdHostData.data(), scoreThresholdHostData.size());
  CHECK_RET(scoresThd != nullptr, return 0);

  // 创建aclTensor: selectedIndices
  ret = CreateAclTensor(outHostData, selectedIndicesShape, &outDeviceAddr, aclDataType::ACL_INT32, &selectedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建attr int: centerPointBox
  int64_t centerPointBox = 0;

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNonMaxSuppression第一段接口
  ret = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxOutputBoxesPerClass, iouThd, scoresThd, centerPointBox, selectedIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppressionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnNonMaxSuppression第二段接口
  ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selectedIndicesShape);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(boxes);
  aclDestroyTensor(scores);
  aclDestroyIntArray(maxOutputBoxesPerClass);
  aclDestroyFloatArray(iouThd);
  aclDestroyFloatArray(scoresThd);
  aclDestroyTensor(selectedIndices);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(boxesDeviceAddr);
  aclrtFree(scoresDeviceAddr);
  aclrtFree(maxSizePerClassDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

```
