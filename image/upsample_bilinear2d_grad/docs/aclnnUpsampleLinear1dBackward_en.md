# aclnnUpsampleLinear1dBackward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_bilinear2d_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     √    |

## Function

- API description: Performs backpropagation of [aclnnUpsampleLinear1d](../../upsample_linear1d/docs/aclnnUpsampleLinear1d_en.md).

- Formula:
  - Core forward algorithm logic:
    1. Scale the destination image to the same size as the source image.
    2. Calculate points of the scaled destination image and points of adjacent source images.
    3. Calculate the weights from the adjacent points to the target points, and multiply and accumulate the weights to obtain the target point values.
  - Calculation logic:
    An image can be scaled by corner alignment (that is, based on the center point of the pixels in the upper left corner of the source image) or edge alignment (that is, based on the vertex in the upper left corner and two edges of the source image). The two modes differ in the scaling factor and coordinates. Then:

    $$
    scale =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales & alignCorners=false\&scales>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point **p** (x, y) in a direction of the output, a point mapped back to the source image is denoted as **q** (x', y'). Then:

    $$
    x' =\begin{cases}
    x * scale & alignCorners=true \\
    MAX(0,{(x+0.5)*scale-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Denoted:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

    - Then:

      $$
      {V(p_{x})} = {V(p_{x0})} * {lambda_{0}} * {lambdb_{0}}+ {V(p_{x1})} * {lambda_{1}} * {lambdb_{0}}
      $$

    - Assume that the output image **out** $(x)$ for forward interpolation is affected by the original image **input** $(x_i)$. Then:
  
      $$
      gradInput(x_i) += gradOut(x) * lambd(x_i)
      $$

## Prototype

Each operator has [two-phase API](./../../../docs/en/context/two_phase_api.md) calls. First, **aclnnUpsampleLinear1dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnUpsampleLinear1dBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnUpsampleLinear1dBackwardGetWorkspaceSize(
  const aclTensor   * gradOut, 
  const aclIntArray * outputSize, 
  const aclIntArray * inputSize, 
  bool                alignCorners, 
  double              scales, 
  aclTensor         * out, 
  uint64_t          * workspaceSize, 
  aclOpExecutor    ** executor)
```

```Cpp
aclnnStatus aclnnUpsampleLinear1dBackward(
  void          * workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor * executor, 
  aclrtStream     stream)
```

## aclnnUpsampleLinear1dGetWorkspaceSize

- **Parameters:**

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
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut</td>
      <td>Input</td>
      <td>Input tensor for upsampling, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The ND format is processed as NCL format by default.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND and NCL</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>Input</td>
      <td>Spatial size of the input `gradOut` in the L dimension, corresponding to `outputSize` in the formula.</td>
      <td>The size is 1, and the value is greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>Input</td>
      <td>Spatial size of the output out in the N, C, and L dimensions, corresponding to `inputSize` in the formula.</td>
      <td>The size is 3, and each element is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners</td>
      <td>Input</td>
      <td>Bool parameter, corresponding to `alignCorners` in the formula.</td>
      <td><ul><li>If True, the input and output tensors are aligned by the center points of their corner pixels, preserving the values at the corner pixels.
      <li>If False, the input and output tensors are aligned by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary values.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Multiplier for the output out in the L dimension, corresponding to `scales` in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor after sampling, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The output shape must be 3D. The data type and format must be the same as those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCL</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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
  </tbody>
  </table>

  - <term>Atlas training products</term>:

    The data types of the input parameter `gradOut` and output parameter `out` must be FLOAT32 or FLOAT16.
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    Input parameter `gradOut`: If the shape of gradOut is different from that of inputSize, the data type can only be FLOAT32 or FLOAT16.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
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
      <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="13">161002</td>
    </tr>
    <tr>
      <td>The data type of gradOut is not supported.</td>
    </tr>
    <tr><td>The data types of gradOut and out are inconsistent.</td>
    </tr>
    <tr><td>The shape of gradOut is not 3D.</td>
    </tr>
    <tr><td>The size of outputSize is not 1.</td>
    </tr>
    <tr><td>The value of an element of outputSize is less than 1.</td>
    </tr>
    <tr><td>The size of inputSize is not 3.</td>
    </tr>
    <tr><td>The value of an element of inputSize is less than 1.</td>
    </tr>
    <tr><td>The sizes of gradOut and inputSize in the N and C dimensions are inconsistent.</td>
    </tr>
    <tr><td>The size of gradOut in the L dimension is different from that of outputSize[0].</td>
    </tr>
    <tr><td>The dimension sizes of the N or C axes of gradOut and out are different.</td>
    </tr>
    <tr><td>The size of the shape of out in each dimension is different from that of the corresponding element in inputSize.</td>
    </tr>
  </tbody></table>

## aclnnUpsampleLinear1dBackward

- **Parameters:**

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnUpsampleLinear1dBackwardGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

## Constraints

- The shape constraints of `gradOut` and `out` are as follows:
  - The value of each dimension is less than or equal to 2^20.
  - The N and C axes of `out` must be the same as those of `gradOut`.
  - The memory usage must be less than 60 GB. The memory size can be calculated according to the following formula:

    $$
    (gradOut\_L + out\_L + out\_L) * N * C  * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

    Where,
    - **N** indicates the N axis of the input and output.
    - **C** indicates the C axis of the input and output.
  - N * C < 2^31
- If the data formats of the input parameter `gradOut` and output parameter `out` are not NCL or ND, the input data is processed as NCL format by default.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The upscaling factor for input data must be less than or equal to 500, that is, $outputSize[0]/(length L of output shape) ≤ 500$.
- Either **outputSize** or **scales** must be used.
  - If **alignCorners** is set to **True**:
    - If **outputSize** is equal to 1, the value of **scales** is 0.
    - In other cases, the values of the input parameters **inputSize** and **outputSize** are used, and $scales = (inputSize\_L – 1)/(outputSize – 1)$. 
  - If **alignCorners** is set to **False**:
    - If the value of **scales** is less than or equal to 0, the value of **outputSize** is used, that is, $scales = (inputSize\_L / outputSize)$.
    - If the value of **scales** is greater than 0 and $outputSize = [floor(inputSize\_L * scales)]$ is not met, the value of **outputSize** is used. Otherwise, the value of **scales** is used.
- Deterministic computing:
  - **aclnnUpsampleLinear1dBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_cv.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_linear_1d_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {1, 1, 3};
    std::vector<int64_t> outShape = {1, 1, 6};
    void *selfDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> outArraySize = {3};
    const aclIntArray *outputSize = aclCreateIntArray(outArraySize.data(), outArraySize.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    std::vector<int64_t> inputArraySize = {1, 1, 6};
    const aclIntArray *inputSize = aclCreateIntArray(inputArraySize.data(), inputArraySize.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnUpsampleLinear1dBackward.
    ret = aclnnUpsampleLinear1dBackwardGetWorkspaceSize(
        self, outputSize, inputSize, true, 1.1, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleLinear1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the computed workspaceSize.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleLinear1dBackward.
    ret = aclnnUpsampleLinear1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleLinear1dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyIntArray(outputSize);
    aclDestroyIntArray(inputSize);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
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
