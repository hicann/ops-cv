# aclnnNonMaxSuppression

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/non_max_suppression_v6)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Operator description: Deletes bounding boxes whose scores are less than **scoreThreshold** and filters out previously selected bounding boxes with high IOU.

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNonMaxSuppressionGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNonMaxSuppression** is called to perform computation.

- `aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, aclIntArray *maxOutputBoxesPerClass, aclFloatArray *iouThreshold, aclFloatArray *scoreThreshold, int32_t centerPointBox, aclTensor *selectedIndices, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnNonMaxSuppression(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNonMaxSuppressionGetWorkspaceSize

- **Parameters:**

  - **boxes** (aclTensor*, compute input): aclTensor on the device. The data type can be FLOAT or FLOAT16. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. The shape is [num_batches, spatial_dimension, 4].

  - **scores** (aclTensor*, compute input): aclTensor on the device. The data type can be FLOAT or FLOAT16. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. The shape is [num_batches, num_classes, spatial_dimension].

  - **maxOutputBoxesPerClass** (aclIntArray*, compute input): maximum number of bounding boxes selected for each class in each batch. aclIntArray on the host. The data type can be INT32. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. The upper limit is 700.

  - **iouThreshold** (aclFloatArray*, compute input): aclFloatArray on the host. The data type can be FLOAT. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. Value range: [0, 1]. It specifies the threshold of IOU to determine whether bounding boxes overlap too much.

  - **scoreThreshold** (aclFloatArray*, compute input): aclFloatArray on the host. The data type can be FLOAT. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. Value range: [0, 1]. It specifies the threshold for determining when to remove a bounding box based on the score.

  - **centerPointBox** (int, compute input): integer on the host. The data type can be INT32. Value range: [0, 1]. It determines the bounding box format. If the value is **0**, this parameter is mainly used for TensorFlow models. The data is provided in the (y1, x1, y2, x2) format, where **(y1, x1)** and **(y2, x2)** are the coordinates of the bounding boxes on the diagonal. Ensure x1 < x2 and y1 < y2. If the value is **1**, this parameter is mainly used for PyTorch models. The data is provided in the (x_center, y_center, width, height) format.

  - **selectedIndices** (aclTensor*, compute output): aclTensor on the device. The data type can be INT32. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format ND](../../../docs/en/context/data_formats.md) is supported. The shape is [num_selected_indices, 3]. The data is provided in the [batch_index, class_index, box_index] format.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  361001 (ACLNN_ERR_RUNTIME_ERROR): 1. The API is not supported by the current product.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input boxes, scores, and out are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of boxes, scores, and maxOutputBoxesPerClass are not supported.
                                    2. The data format of boxes, scores, or selectedIndices is not supported.
                                    3. The shape of boxes or scores is not supported.
                                        (1) boxes and scores must be 3-dimensional.
                                        (2) Dimension 0 of boxes must be equal to that of scores.
                                        (3) Dimension 1 of boxes must be equal to dimension 2 of scores.
                                        (4) Dimension 2 of boxes must be 4.
                                    4. The values of iouThreshold, scoreThreshold, centerPointBox, and maxOutputBoxesPerClass are not supported.
  ```

## aclnnNonMaxSuppression

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnNonMaxSuppressionGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

## Constraints

1. The upper limit of **maxOutputBoxesPerClass** is 700. The data types of the input parameters **boxes** and **scores** must be the same.
2. In the FLOAT16 scenario, a computation error may be introduced when operators are sorted and computed to compare with the benchmark.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_cv.md).
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
  // (Fixed writing) Initialize resources.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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

  // Create an aclTensor: boxes.
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an aclTensor: scores.
  ret = CreateAclTensor(scoresHostData, scoresShape, &scoresDeviceAddr, aclDataType::ACL_FLOAT, &scores);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an AclIntArray: maxOutputBoxesPerClass.
  ret = CreateAclIntArray(maxOutputBoxesPerClassHostData, &maxSizePerClassDeviceAddr, &maxOutputBoxesPerClass);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an AclFloatArray: iouThreshold.
  iouThd = aclCreateFloatArray(iouThresholdHostData.data(), iouThresholdHostData.size());
  CHECK_RET(iouThd != nullptr, return 0);

  // Create an AclFloatArray: scoresThreshold.
  scoresThd = aclCreateFloatArray(scoreThresholdHostData.data(), scoreThresholdHostData.size());
  CHECK_RET(scoresThd != nullptr, return 0);

  // Create an aclTensor: selectedIndices.
  ret = CreateAclTensor(outHostData, selectedIndicesShape, &outDeviceAddr, aclDataType::ACL_INT32, &selectedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an attr int: centerPointBox.
  int64_t centerPointBox = 0;

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNonMaxSuppression.
  ret = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxOutputBoxesPerClass, iouThd, scoresThd, centerPointBox, selectedIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppressionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnNonMaxSuppression.
  ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selectedIndicesShape);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(boxes);
  aclDestroyTensor(scores);
  aclDestroyIntArray(maxOutputBoxesPerClass);
  aclDestroyFloatArray(iouThd);
  aclDestroyFloatArray(scoresThd);
  aclDestroyTensor(selectedIndices);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
