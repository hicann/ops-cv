# aclnnUpsampleNearest2dBackward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_nearest2d_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     √    |

## Function

- API description: Performs backpropagation of [aclnnUpsampleNearest2dV2](./aclnnUpsampleNearest2dV2.md).

- Formulas:

  $$
  gradInput(N, C, H, W) += gradOutput( N, C, ceil ( scalesH * H ),  ceil ( scalesW * W ))
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnUpsampleNearest2dBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnUpsampleNearest2dBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut, 
  const aclIntArray *outputSize, 
  const aclIntArray *inputSize, 
  double             scalesH, 
  double             scalesW, 
  aclTensor*         gradInput, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleNearest2dBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnUpsampleNearest2dBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient tensor for backpropagation, corresponding to gradOutput in the formula.</td>
      <td><ul><li>The H and W axes do not support empty tensors. </li><li>NCHW and NHWC are supported. </li><li>The input shape must be 4D.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW and NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize (aclIntArray*)</td>
      <td>Input</td>
      <td>Spatial size of the input gradOut in the H and W dimensions.</td>
      <td>The size is 2, and each element is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize (aclIntArray*)</td>
      <td>Input</td>
      <td>Space size of gradInput.</td>
      <td><ul><li>The size is 4, and the last two elements are greater than 0. </li><li>If the input gradOut is in NCHW format, it indicates the spatial size of the output gradInput in the N, C, H, and W dimensions. If the input gradOut is in NHWC format, it indicates the spatial size of the output gradInput in the N, H, W, and C dimensions.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH (double)</td>
      <td>Input</td>
      <td>Multiplier for the output gradInput in the height dimension, corresponding to scalesH in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW (double)</td>
      <td>Input</td>
      <td>Multiplier for the output gradInput in the width dimension, corresponding to scalesW in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor for backpropagation, corresponding to gradInput in the formula.</td>
      <td><ul><li>The H and W axes do not support empty tensors. </li><li>The data type and format are the same as those of the input parameter gradOut. </li><li>The N and C axes of the shape are the same as those of the input parameter gradOut.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW and NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>:
  
    The data types of the input parameter gradOut and output parameter gradInput support only FLOAT16.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
    </tr>
    <tr>
      <td>The data type of gradOut is not supported.</td>
    </tr>
    <tr><td>The data types of gradOut and gradInput are inconsistent.</td>
    </tr>
    <tr><td>The shape of gradOut is not 4D.</td>
    </tr>
    <tr><td>The size of outputSize is not 2.</td>
    </tr>
    <tr><td>The value of an element of outputSize is less than 1.</td>
    </tr>
    <tr><td>The size of inputSize is not 4.</td>
    </tr>
    <tr><td>One of the last two elements of inputSize is less than 1.</td>
    </tr>
    <tr><td>The sizes of gradOut in the H and W dimensions are inconsistent with those of outputSize[0] and outputSize[1].</td>
    </tr>
    <tr><td>The dimension sizes of the N or C axes of gradOut and gradInput are different.</td>
    </tr>
  </tbody></table>

## aclnnUpsampleNearest2dBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnUpsampleNearest2dBackwardGetWorkspaceSize.</td>
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

- The inputSize, outputSize, scalesH, and scalesW parameters must meet the following restrictions:

  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- Deterministic computation:
  - **aclnnUpsampleNearest2dBackward** defaults to a deterministic implementation.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_2d_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> gradOutShape = {2, 2, 3, 3};
    std::vector<int64_t> gradInputShape = {2, 2, 1, 1};
    void* gradOutDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
    std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
    std::vector<int64_t> outputSizeData = {3, 3};
    std::vector<int64_t> inputSizeData = {2, 2, 1, 1};
    double scalesH = 0.0;
    double scalesW = 0.0;
    // Create a gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    const aclIntArray* inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnUpsampleNearest2dBackward.
    ret = aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
        gradOut, outputSize, inputSize, scalesH, scalesW, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnUpsampleNearest2dBackward.
    ret = aclnnUpsampleNearest2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest2dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
