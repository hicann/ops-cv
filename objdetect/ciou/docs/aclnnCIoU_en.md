# aclnnCIoU

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

* Description: A loss function used for bounding box regression. Based on the IoU, the center point distance, aspect ratio, and overlapping area are considered to comprehensively measure the difference between the predicted box and the real box.
* Formulas:

$$
CIoU = IoU - \frac{\rho^2(b^p, b^g)}{c^2} - \alpha v \\
v = \frac{4}{\pi^2}(arctan(\frac{w^g}{h^g}) - arctan(\frac{w^p}{h^p}))^2 \\
\alpha = \frac{v}{1 - IoU + v} \\
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCIoUGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCIoU` is called to perform computation.

```cpp
aclnnStatus aclnnCIoUGetWorkspaceSize(
  const aclTensor   *bBoxes,
  const aclTensor   *gtBoxes,
  bool               trans,
  bool               isCross,
  const char        *mode,
  aclTensor         *overlap,
  aclTensor         *atanSub,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor);
```

```cpp
aclnnStatus aclnnCIoU(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream);
```

## aclnnCIoUGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1533px"><colgroup>
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 290px">
  <col style="width: 252px">
  <col style="width: 128px">
  <col style="width: 149px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Format Type</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>bBoxes</td>
      <td>Input</td>
      <td>Prediction bounding box.</td>
      <td>2D tensor with shape [4, M].</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gtBoxes</td>
      <td>Input</td>
      <td>Bounding box in the truth value.</td>
      <td>2D tensor of shape [4, N].</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Input</td>
      <td>Format of the bounding box.</td>
      <td>true: The input format is [x, y, w, h].<br>false: The input format is [x0, y0, x1, y1].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isCross</td>
      <td>Input</td>
      <td>Whether to perform intersection operation between bBoxes and gtBoxes.</td>
      <td>true: The output shape is [M, N].<br>false: The output shape is [1, N].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Input</td>
      <td>Calculation mode, which can be "iou" or "iof".</td>
      <td>-</td>
      <td>CHAR*</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>Output</td>
      <td>Intersection over union or foreground intersection ratio calculated based on two inputs.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>atanSub</td>
      <td>Output</td>
      <td>Difference between two arctan values during calculation.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_Return_Code.md).

   The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
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
      <td>bBoxes, gtBoxes, overlap, or atanSub is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>bBoxes, gtBoxes, overlap, or atanSub is not two-dimensional.</td>
    </tr>
    <tr>
      <td>bBoxes, gtBoxes, overlap, or atanSub has inconsistent data types.</td>
    </tr>
    <tr>
      <td>bBoxes, gtBoxes, overlap, or atanSub has unsupported data types or formats.</td>
    </tr>
    <tr>
      <td>The first dimension of bBoxes or gtBoxes is not 4.</td>
    </tr>
    <tr>
      <td>The second dimension of bBoxes or gtBoxes is not a multiple of 1024.</td>
    </tr>
    <tr>
      <td>The first dimension of overlap or atanSub is not 1.</td>
    </tr>
    <tr>
      <td>The second dimensions of bBoxes, gtBoxes, overlap, and atanSub are not equal.</td>
    </tr>
    <tr>
      <td>isCross is not false.</td>
    </tr>
    <tr>
      <td>mode is not "iou" or "iof".</td>
    </tr>
  </tbody>
  </table>

## aclnnCIoU

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 319px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnCIoUGetWorkspaceSize.</td>
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
  - The aclnnCIoU is implemented in deterministic mode by default.
- If the input format is [x0, y0, x1, y1], (x0, y0) and (x1, y1) indicate the upper left corner and lower right corner of the bounding box, respectively. The following condition must be met: x1 > x0, y1 > y0
- M and N must be the same.
- Currently, isCross can only be set to false.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ciou.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2.Construct the inputs and outputs based on the API definition.
  // input
  std::vector<float> bBoxesHostData(4096, 1);
  std::vector<float> gtBoxesHostData(4096, 2);
  std::vector<float> overlapHostData(1024, 0);
  std::vector<float> atanSubHostData(1024, 0);
  std::vector<int64_t> bBoxesShape = {4, 1024};
  std::vector<int64_t> gtBoxesShape = {4, 1024};
  std::vector<int64_t> overlapShape = {1, 1024};
  std::vector<int64_t> atanSubShape = {1, 1024};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  void* atanSubDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;
  aclTensor* atanSub = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(atanSubHostData, atanSubShape, &atanSubDeviceAddr, aclDataType::ACL_FLOAT, &atanSub);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  bool trans = false;
  bool isCross = false;
  const char* mode = "iou";

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  ret = aclnnCIoUGetWorkspaceSize(bBoxes, gtBoxes, trans, isCross, mode, overlap, atanSub, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnCIoUGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnCIoU
  ret = aclnnCIoU(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnCIoU failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto overlapSize = GetShapeSize(overlapShape);
  std::vector<float> overlapData(overlapSize, 0);
  ret = aclrtMemcpy(overlapData.data(), overlapData.size() * sizeof(overlapData[0]), overlapDeviceAddr,
                    overlapSize * sizeof(overlapData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy overlapData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < overlapSize; i++) {
    LOG_PRINT("overlap[%ld] is: %f\n", i, overlapData[i]);
  }

  auto atanSubsize = GetShapeSize(atanSubShape);
  std::vector<float> atanSubData(atanSubsize, 0);
  ret = aclrtMemcpy(atanSubData.data(), atanSubData.size() * sizeof(atanSubData[0]), atanSubDeviceAddr,
                    atanSubsize * sizeof(atanSubData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy atanSubData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < atanSubsize; i++) {
    LOG_PRINT("atanSub[%ld] is: %f\n", i, atanSubData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);
  aclDestroyTensor(atanSub);

  // 7. Free device resources.
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);
  aclrtFree(atanSubDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
