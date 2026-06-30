# aclnnGridSampler3DBackward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/image/grid_sampler3_d_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     √    |

## Function

- API description: Performs backpropagation of [aclnnGridSampler3D](../../grid_sample/docs/aclnnGridSampler3D_en.md) to compute the gradients of tensors **input** and **grid**.
- Formula:

  - Computing process:
    1. The coordinates mapped to **input** are calculated based on the (x, y, z) values stored in **grid**. The coordinates are related to **alignCorners** and **paddingMode**.
    2. The output value is calculated by using the input **interpolationMode** (bilinear, nearest, or bicubic).
    3. The final **dx** and **dgrid** results are calculated based on the gradient value stored in **grad** multiplied by the weight of the corresponding point.
  
  - Where,
  
      The dimensions of **grad**, **input**, **grid**, **dx**, and **dgrid** are as follows:
  
      $$
      grad: (N, C, D_{out}, H_{out}, W_{out})\\
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      dx: (N, C, D_{in}, H_{in}, W_{in})\\
      dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
      $$
  
      **grad**, **input**, **grid**, **dx**, and **dgrid** have the same **N**. **grad**, **input**, and **dx** have the same **C**. **$D_{in}$**, **$H_{in}$**, and **$W_{in}$** in **input** and **dx** are the same. **$D_{out}$**, **$H_{out}$**, and **$W_{out}$** in **grad**, **grid**, and **dgrid** are the same. The size of the last dimension of **grid** is **3**, indicating that the input pixel location is (x, y, z). The value ranges of **x**, **y**, and **z** are normalized to [-1, 1].

    - Out-of-bound coordinates are processed based on the padding mode.
  
      - **paddingMode="zeros"**: Use 0 for out-of-bound locations.
      - **paddingMode="border"**: Use border values for out-of-bound locations.
      - **paddingMode="reflection"**: Use symmetric values of the border values for out-of-bound locations.
  
    - During input sampling, the processing varies according to **interpolationMode**.
  
      - **interpolationMode="bilinear"**: Use the weighted average value of the 8 coordinates around (x, y, z) in the **input**.
      - **interpolationMode="nearest"**: Use the nearest coordinates to (x, y, z) in the **input**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGridSampler3DBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGridSampler3DBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(
  const aclTensor*    gradOutput,
  const aclTensor*    input,
  const aclTensor*    grid,
  int64_t             interpolationMode,
  int64_t             paddingMode,
  bool                alignCorners,
  const aclBoolArray* outputMask,
  aclTensor*          inputGrad,
  aclTensor*          gridGrad,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnGridSampler3DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler3DBackwardGetWorkspaceSize

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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Output gradient of the upper layer during backpropagation, corresponding to `grad` in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `input`. </li><li>When the data type is DOUBLE, the data format cannot be NDHWC.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW and NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Input tensor for backpropagation, corresponding to `input` in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When the data type is DOUBLE, the data format cannot be NDHWC. </li><li>The values on the N and C axes of `input` must be the same as those of `gradOutput`. The D, H, and W values of `input` cannot be 0.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW and NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>Input</td>
      <td>Tensor that uses pixel locations, corresponding to `grid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `input`. </li><li>The values on the N, D, H, and W axes of `grid` must be the same as those of `gradOutput`. The value on the C axis must be 3.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode</td>
      <td>Input</td>
      <td>Interpolation mode, corresponding to `interpolationMode` in the formula description.</td>
     <td>The value can be 0 (bilinear) or 1 (nearest).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode</td>
      <td>Input</td>
      <td>Padding mode. If there are grid values beyond the range of [-1, 1], the corresponding output values are processed according to the mode defined by paddingMode. This corresponds to `paddingMode` in the formula description.</td>
      <td>The value can be 0 (zeros), 1 (border), or 2 (reflection).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners</td>
      <td>Input</td>
      <td>Alignment mode between the feature map coordinates and feature values, corresponding to `alignCorners` in the formula description.</td>
      <td>If True, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. If False, they are instead considered as referring to the corner points of the input's corner pixels.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>Output mask.</td>
      <td>If outputMask[0] is True or False, inputGrad is or is not obtained. If outputMask[1] is True or False, gridGrad is or is not obtained.</li></ul></td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputGrad</td>
      <td>Output</td>
      <td>Output gradient for backpropagation, corresponding to `dx` in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type, data format, and shape must be the same as those of `input`. </li><li>When the data type is DOUBLE, the data format cannot be NDHWC.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW and NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad</td>
      <td>Output</td>
      <td>Grid gradient, corresponding to `dgrid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type, data format, and shape must be the same as those of `grid`.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NDHWC</td>
      <td>5</td>
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
  
    The data types of `gradOutput`, `input`, `grid`, `inputGrad`, and `gridGrad` do not support BFLOAT16.

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
      <td>The passed gradOutput, input, grid, inputGrad, or gridGrad is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of gradOutput, input, grid, inputGrad, or gridGrad is not supported or their data types are different.</td>
    </tr>
    <tr>
      <td>The value of interpolationMode or paddingMode is not supported.</td>
    </tr>
    <tr>
      <td>The dimensions of gradOutput, input, grid, inputGrad, and gridGrad do not match.</td>
    </tr>
  </tbody></table>

## aclnnGridSampler3DBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGridSampler3DBackwardGetWorkspaceSize.</td>
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

- Deterministic computing:
  - **aclnnGridSampler3DBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computing.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_cv.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler3d_backward.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
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
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    aclBoolArray* outputMask = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 1, 2, 2};
    std::vector<int64_t> inputShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridShape = {1, 1, 2, 2, 3};
    std::vector<int64_t> inputGradShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridGradShape = {1, 1, 2, 2, 3};
    void* gradOutputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* gridGradDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* gridGrad = nullptr;

    std::vector<float> gradOutputHostData = {1, 1, 1, 1};
    std::vector<float> inputHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    };
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    const uint64_t kNum = 2U;
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum);

    // Create a gradOutput aclTensor.
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a grid aclTensor.
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an inputGrad aclTensor.
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gridGrad aclTensor.
    ret = CreateAclTensor(gridGradHostData, gridGradShape, &gridGradDeviceAddr, aclDataType::ACL_FLOAT, &gridGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGridSampler3DBackward.
    ret = aclnnGridSampler3DBackwardGetWorkspaceSize(
        gradOutput, input, grid, interpolationMode, paddingMode, alignCorners, outputMask, inputGrad, gridGrad,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGridSampler3DBackward.
    ret = aclnnGridSampler3DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto inputGradSize = GetShapeSize(inputGradShape);
    std::vector<float> inputGradResultData(inputGradSize, 0);
    ret = aclrtMemcpy(
        inputGradResultData.data(), inputGradResultData.size() * sizeof(inputGradResultData[0]), inputGradDeviceAddr,
        inputGradSize * sizeof(inputGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy inputGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < inputGradSize; i++) {
        LOG_PRINT("inputGradResultData[%ld] is: %f\n", i, inputGradResultData[i]);
    }

    auto gridGradSize = GetShapeSize(gridGradShape);
    std::vector<float> gridGradResultData(gridGradSize, 0);
    ret = aclrtMemcpy(
        gridGradResultData.data(), gridGradResultData.size() * sizeof(gridGradResultData[0]), gridGradDeviceAddr,
        gridGradSize * sizeof(gridGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gridGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < gridGradSize; i++) {
        LOG_PRINT("gridGradResultData[%ld] is: %f\n", i, gridGradResultData[i]);
    }

    // 6. Release aclTensor and aclBoolArray. Modify the configuration based on the API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(gridGrad);
    aclDestroyBoolArray(outputMask);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(gridGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
