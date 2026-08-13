# aclnnIou

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/iou_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Operator description: Calculates the intersection over union (IOU) or intersection over foreground (IOF) of two sets of input bounding boxes to evaluate the overlap between the predicted bounding box (bBox) and the ground truth bounding box (gtBox).
- Formulas:

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2} 
  $$

  **Area_1** indicates the area of the **bBox**, **Area_2** indicates the area of the **gtBox**, and **Area_3** indicates the area of the overlapping part. For the definitions of **x** and **y**, see the parameter description.

  $$
 are a_1 = (X_1 - X_0)(Y_1 - Y_0) \\
 are a_2 = (X_3 - X_2)(Y_3 - Y_2) \\
 are a_3 = max( min(X_1, X_3) - max(X_0, X_2), 0 ) * max( min(Y_1, Y_3) - max(Y_0, Y_2), 0 )
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnIouGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnIou** is called to perform computation.

```Cpp
aclnnStatus aclnnIouGetWorkspaceSize(
  const aclTensor*        bBoxes, 
  const aclTensor*        gtBoxes, 
  const char*             mode, 
  float                   eps, 
  bool                    aligned, 
  aclTensor*              overlap, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnIou(
  void*                   workspace, 
  uint64_t                workspaceSize,  
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnIouGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 400px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 120px">
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
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">bBoxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Predicted bounding box.</td>
      <td class="tg-0pky"><ul><li>The m in shape indicates the number of bounding boxes. </li><li>The 4 in shape indicates [x0, y0, x1, y1], where (x0, y0) and (x1, y1) indicate the upper left corner and lower right corner of the bounding box, respectively. The following condition must be met: x1 > x0, y1 > y0.</li></ul></td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">The value is (m, 4).</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gtBoxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Ground truth bounding box.</td>
      <td class="tg-0pky"><ul><li>n indicates the number of bounding boxes. </li><li>4 indicates [x2, y2, x3, y3], where (x2, y2) and (x3, y3) indicate the upper left corner and lower right corner of the bounding box, respectively. The value must meet the x3 > x2, y3 > y2 condition. </li><li>The data type must be the same as that of bBoxes.</li></ul></td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">The value is (n, 4).</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mode (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Selects the calculation mode, which can be "iou" or "iof".</td>
      <td class="tg-0lax"><ul><li>"iou": Calculates the intersection over union. </li><li>"iof": Calculates the intersection over foreground.</li></ul></td>
      <td class="tg-0lax">String</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps (float)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">To prevent division by zero, eps is added to the length and width when calculating the area.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">aligned (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether the shapes of the two inputs are the same.</td>
      <td class="tg-0lax"><ul><li>True: The shapes of bBoxes and gtBoxes are both (m, 4), and the output shape is (m, 1). </li><li>False: The shapes of bBoxes and gtBoxes are (m, 4) and (n, 4) respectively, and the output shape is (m, n).</li></ul></td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">overlap (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Intersection over union (IoU) or foreground intersection ratio calculated based on two inputs.</td>
      <td class="tg-0lax">The data type must be the same as that of bBoxes.</td>
      <td class="tg-0lax">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">The value is (m, n) or (m, 1).</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device.</td>
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

  - <term>Atlas inference products</term>: The data type BFLOAT16 is not supported.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_Return_Code.md).

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input bBoxes, gtBoxes, and output overlap are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>bBoxes, gtBoxes, and overlap are not two-dimensional.</td>
    </tr>
    <tr>
      <td>bBoxes, gtBoxes, and overlap have inconsistent data types.</td>
    </tr>
    <tr>
      <td>bBoxes, gtBoxes, and overlap have unsupported data types and formats.</td>
    </tr>
    <tr>
      <td>The second dimension of bBoxes or gtBoxes is not 4.</td>
    </tr>
    <tr>
      <td>When aligned is true, the first dimension of bBoxes is different from that of gtBoxes.</td>
    </tr>
    <tr>
      <td>When aligned is true, the second dimension of overlap is not 1.</td>
    </tr>
    <tr>
      <td>mode is not "iou" or "iof".</td>
    </tr>
    <tr>
      <td>eps is less than 0.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by unsupported input data or attribute specifications.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An error occurred when the NPU runtime API is called, for example, due to unsupported SocVersion.</td>
    </tr>
  </tbody>
  </table>

## aclnnIou

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnIouGetWorkspaceSize.</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_Return_Code.md).

## Constraints

- Deterministic computation:
  - **aclnnIou** defaults to a deterministic implementation.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_iou.h"

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

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2.Construct the inputs and outputs based on the API definition.
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  ret = aclnnIouGetWorkspaceSize(bBoxes, gtBoxes, mode, eps, aligned, overlap, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnIouGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnIou
  ret = aclnnIou(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnIou failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(overlapShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), overlapDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);

  // 7. Free device resources.
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
