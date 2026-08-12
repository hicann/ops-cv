# aclnnNonMaxSuppression

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/non_max_suppression_v6)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Deletes the bounding boxes whose scores are less than scoreThreshold and selects the boxes that have a high overlap with the previously selected boxes (high IOU).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNonMaxSuppressionGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNonMaxSuppression** is called to perform computation.

```Cpp
aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(
  const aclTensor*        boxes, 
  const aclTensor*        scores, 
  aclIntArray*            maxOutputBoxesPerClass, 
  aclFloatArray*          iouThreshold, 
  aclFloatArray*          scoreThreshold, 
  int32_t                 centerPointBox, 
  aclTensor*              selectedIndices, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnNonMaxSuppression(
  void*                   workspace, 
  uint64_t                workspaceSize, 
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnNonMaxSuppressionGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 300px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 224px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">boxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, spatial_dimension, 4]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">scores (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, num_classes, spatial_dimension]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">maxOutputBoxesPerClass (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Maximum number of selected bounding boxes for each class in each batch.</td>
      <td class="tg-0lax">The maximum value is 700.</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">iouThreshold (aclFloatArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Threshold for determining whether the bounding box overlaps too much with the IOU.</td>
      <td class="tg-0lax">The value range is [0, 1].</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">scoreThreshold (aclFloatArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Threshold for determining when to remove a bounding box based on the score.</td>
      <td class="tg-0lax">The value range is [0, 1].</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">centerPointBox (int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Determines the format of the bounding box.</td>
      <td class="tg-0lax"><ul><li>Value range: [0, 1]. </li><li>When the value is 0, it is mainly used for TensorFlow models. The data is provided in the format of (y1, x1, y2, x2), where (y1, x1) and (y2, x2) are the coordinates of the diagonal corners of the bounding box. You need to ensure that x1 < x2 and y1 < y2 are correct. </li><li>When the value is 1, it is mainly used for PyTorch models. The data is provided in the format of (x_center, y_center, width, height).</li></ul></td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">selectedIndices (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output tensor.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">[num_selected_indices, 3]<br>The data is provided in the format of [batch_index, class_index, box_index].</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Operator executor, including the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 290px">
  <col style="width: 134px">
  <col style="width: 844px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current product is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input boxes, scores, and selectedIndices are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of boxes, scores, and maxOutputBoxesPerClass are not supported.</td>
    </tr>
    <tr>
      <td>The data formats of boxes, scores, and selectedIndices are not supported.</td>
    </tr>
    <tr>
      <td>boxes and scores must be 3-dimensional.</td>
    </tr>
    <tr>
      <td>The 0th dimension of boxes must be equal to the 0th dimension of scores.</td>
    </tr>
    <tr>
      <td>The first dimension of boxes must be equal to the second dimension of scores.</td>
    </tr>
    <tr>
      <td>The second dimension of boxes must be equal to 4.</td>
    </tr>
    <tr>
      <td>The values of iouThreshold, scoreThreshold, centerPointBox, and maxOutputBoxesPerClass are not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnNonMaxSuppression

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 170px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnNonMaxSuppressionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. The upper limit of **maxOutputBoxesPerClass** is 700. The data types of the input parameters **boxes** and **scores** must be the same.
2. In the FLOAT16 scenario, a computation error may be introduced when operators are sorted and computed to compare with the benchmark.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Boilerplate) Initialize resources.
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

  // Call `aclCreateIntArray` to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
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
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnNonMaxSuppression.
  ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selectedIndicesShape);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(boxes);
  aclDestroyTensor(scores);
  aclDestroyIntArray(maxOutputBoxesPerClass);
  aclDestroyFloatArray(iouThd);
  aclDestroyFloatArray(scoresThd);
  aclDestroyTensor(selectedIndices);

  // 7. Release device resources. Modify the code based on the API definition.
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
