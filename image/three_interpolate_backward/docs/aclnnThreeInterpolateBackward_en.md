# aclnnThreeInterpolateBackward

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnThreeInterpolateBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnThreeInterpolateBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(
  const aclTensor *grad_x,
  const aclTensor *idx,
  const aclTensor *weight,
  int              m, 
  aclTensor       *grad_y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnThreeInterpolateBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## Function

- Operator description: Performs three-point interpolation calculation based on **grad_x**, **idx**, and **weight** to obtain the gradient **grad_y**.

- Formulas:

  $$
  grad\_y[b,c,idx[b,n,i]] = 
  grad\_y[b,c,idx[b,n,i]] + grad\_x[b,c,n]*weight[b,n,i]\\ i\in[0,2]\ b\in[0,B) \ c\in[0,C) \ n\in[0,N)
  $$

## aclnnThreeInterpolateBackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1556px"><colgroup>
  <col style="width: 171px">
  <col style="width: 121px">
  <col style="width: 271px">
  <col style="width: 331px">
  <col style="width: 224px">
  <col style="width: 103px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0lax">Usage Description</th>
      <th class="tg-0lax">Data Type</th>
      <th class="tg-0lax">Data Format</th>
      <th class="tg-0lax">Dimension (shape)</th>
      <th class="tg-0lax">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">grad_x (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Gradient value of the previous step before backpropagation.</td>
      <td class="tg-0lax">The shape supports (b, c, n).</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">Support NCHW.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">idx (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Index of the three nearest neighbors of the target feature.</td>
      <td class="tg-0lax">The shape supports (b, n, 3). Empty tensors are not supported.</td>
      <td class="tg-0lax">INT32, INT64</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weight (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight of the three nearest neighbors of the target feature.</td>
      <td class="tg-0lax">The shape supports (b, n, 3). Empty tensors are not supported.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">m (int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Minimum size of the input feature dimension, which is used to guide the inference of the grad_y shape during shape inference.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad_y (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Gradient calculation result.</td>
      <td class="tg-0lax">The shape supports (b, c, m).</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">Support NCHW.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, including the operator computation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_Return_Code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  
  <table class="tg" style="undefined;table-layout: fixed; width: 859px"><colgroup>
  <col style="width: 302px">
  <col style="width: 142px">
  <col style="width: 415px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input pointers grad_x, idx, weight, and grad_y are null.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="2">161002</td>
      <td class="tg-0pky">The data types and formats of grad_x, idx, weight, m, and grad_y are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0lax">When grad_x, idx, weight, and grad_y are not null, their data types, formats, or shapes are not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnThreeInterpolateBackward

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnThreeInterpolateBackwardGetWorkspaceSize.</td>
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

- The values in **idx** must be less than m.
- Deterministic computation:
  - **aclnnThreeInterpolateBackward** defaults to a non-deterministic implementation. Enabling deterministic computing is not supported.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Second-phase API of aclnnThreeInterpolateBackward, which is used to perform computation.
*/
aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief First-phase API of aclnnThreeInterpolateBackward, which is used to calculate the workspace size based on the specific computation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *grad_x, const aclTensor *idx,
                                                          const aclTensor *weight, int m, aclTensor *grad_y,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  uint32_t bs = 1;
  uint32_t cs = 2;
  uint32_t ms = 6;
  uint32_t ns = 5;
  uint32_t c0 = 16;

  auto c1 = (cs + c0 - 1) / c0;

  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> gradXShape = {bs, cs, ns, 1}; // nchw
  std::vector<int64_t> idxShape = {bs, ns, 3}; // nd
  std::vector<int64_t> weightShape = {bs, ns, 3}; // nd
  std::vector<int64_t> gradYShape = {bs, cs, ms, 1}; // nchw

  void* gradXDeviceAddr = nullptr;
  void* idxDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradYDeviceAddr = nullptr;

  aclTensor* gradX = nullptr;
  aclTensor* idx = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradY = nullptr;

  std::vector<float> gradXHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint32_t> idxHostData = {0, 3, 2, 3, 5, 1, 0, 3, 0, 0, 3, 5, 0, 4, 1};
  std::vector<float> weightHostData = {3.2081969,-0.86573875,-1.0929844,5.6045847,5.328887,-2.0821328,0.98572457,9.612394,2.216394,
                                       1.8722068,4.2976365,-7.019285,-8.4070425,1.0159919,-0.78814566};
  std::vector<float> gradYHostData = {0};

  ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(idxHostData, idxShape, &idxDeviceAddr, aclDataType::ACL_INT32, &idx, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnThreeInterpolateBackward.
  ret = aclnnThreeInterpolateBackwardGetWorkspaceSize(gradX, idx, weight, ms, gradY, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnThreeInterpolateBackward.
  ret = aclnnThreeInterpolateBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(gradYShape, &gradYDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(gradX);
  aclDestroyTensor(idx);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradY);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradXDeviceAddr);
  aclrtFree(idxDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
