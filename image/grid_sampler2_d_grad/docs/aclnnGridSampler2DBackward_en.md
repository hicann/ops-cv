
# aclnnGridSampler2DBackward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/image/grid_sampler2_d_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     √    |

## Function

- API description: Performs backpropagation of [aclnnGridSampler2D](./aclnnGridSampler2D.md) to compute the gradients of tensors **input** and **grid**.
- Formulas:
  - Computing process:

    1. The coordinates mapped to **input** are calculated based on the (x, y) values stored in **grid**. The coordinates are related to **alignCorners** and **paddingMode**.
    2. Based on the input **interpolationMode**, the bilinear or nearest interpolation mode is used to calculate the weight of the point around the coordinate allocated to the gradient.
    3. The final **dx** and **dgrid** results are calculated based on the gradient value stored in **grad** multiplied by the weight of the corresponding point.
  
  - The values are as follows:
      The dimensions of **grad**, **input**, **grid**, **dx**, and **dgrid** are as follows:

      $$
      grad: (N, C, H_{out}, W_{out})\\
      input: (N, C, H_{in}, W_{in})\\
      grid: (N, H_{out}, W_{out}, 2)\\
      dx: (N, C, H_{in}, W_{in})\\
      dgrid: (N, H_{out}, W_{out}, 2)
      $$
  
      The values of N in grad, input, grid, dx, and dgrid are the same. The values of C in grad, input, and dx are the same. The values of $H_{in}$ and $W_{in}$ in input and dx are the same. The values of $H_{out}$ and $W_{out}$ in grad, grid, and dgrid are the same. The last dimension of grid is 2, indicating that the pixel position information of the input is (x, y). The values of x and y are normalized to the range of [-1, 1]. (-1, 1) indicates the coordinates of the upper left corner, and (1, -1) indicates the coordinates of the lower right corner.

    - Out-of-bound coordinates are processed based on the padding mode.
  
      - **paddingMode="zeros"**: Use 0 for out-of-bound locations.
      - **paddingMode="border"**: Use border values for out-of-bound locations.
      - **paddingMode="reflection"**: Use symmetric values of the border values for out-of-bound locations.
  
    - During input sampling, the processing varies according to **interpolationMode**.
  
      - **interpolationMode="bilinear"**: Use the weighted average value of the 4 coordinates around (x, y) in the input.
      - **interpolationMode="nearest"**: Use the nearest coordinates to (x, y) in the **input**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGridSampler2DBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGridSampler2DBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     input,
  const aclTensor*     grid,
  int64_t              interpolationMode,
  int64_t              paddingMode,
  bool                 alignCorners,
  const aclBoolArray*  outputMask,
  aclTensor*           inputGrad,
  aclTensor*           gridGrad,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnGridSampler2DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler2DBackwardGetWorkspaceSize

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
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Output gradient of the upper layer during backpropagation, corresponding to grad in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>The shape can be (N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor for backpropagation, corresponding to input in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The values on the N and C axes of input must be the same as those of gradOutput. The last two dimensions of input cannot be 0. </li><li>The shape can be (N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>in</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>in</em>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor that uses pixel locations, corresponding to grid in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>The values on the N, H, and W axes of grid must be the same as those of gradOutput. The last dimension of grid must be 2. </li><li>The shape can be (N, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>, 2).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode (int64_t)</td>
      <td>Input</td>
      <td>Interpolation mode, corresponding to interpolationMode in the formula description.</td>
      <td>The value can be 0 (bilinear) or 1 (nearest).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode (int64_t)</td>
      <td>Input</td>
      <td>Padding mode. When the values of (x, y) exceed the sampling range of the input feature map, a specific value is returned. This corresponds to paddingMode in the formula description.</td>
      <td>The value can be 0 (zeros), 1 (border), or 2 (reflection).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners (bool)</td>
      <td>Input</td>
      <td>Alignment mode between the feature map coordinates and feature values, corresponding to alignCorners in the formula description.</td>
      <td>If true, the feature values are located in the pixel center. If false, the feature values are located at the corner points of the corner pixels.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask (aclBoolArray*)</td>
      <td>Input</td>
      <td>Output mask.</td>
      <td>If outputMask[0] is true or false, inputGrad is or is not obtained. If outputMask[1] is true or false, gridGrad is or is not obtained.</td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputGrad (aclTensor*)</td>
      <td>Output</td>
      <td>Output gradient for backpropagation, corresponding to dx in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of input. </li><li>The shape is the same as that of the input.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad (aclTensor*)</td>
      <td>Output</td>
      <td>Grid gradient, corresponding to dgrid in the formula description.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of input. </li><li>The shape is the same as that of the grid.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
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

  - <term>Atlas training products</term>:
  
    The data types of gradOutput, input, grid, inputGrad, and gridGrad do not support BFLOAT16 or DOUBLE.

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
      <td>The passed gradOutput, input, grid, inputGrad, or gridGrad is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type of gradOutput, input, grid, inputGrad, or gridGrad is not supported.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput, input, or grid is not 4D.</td>
    </tr>
    <tr>
      <td>The value of interpolationMode or paddingMode is not supported.</td></tr>
    <tr>
      <td>The first dimension of input has a different value from that of grid and gradOutput.</td>
    </tr>
    <tr>
      <td>The second dimension of input has a different value from that of gradOutput.</td>
    </tr>
    <tr>
      <td>The second dimension of grid has a different value from the third dimension of gradOutput.</td>
    </tr>
    <tr>
      <td>The third dimension of grid has a different value from the fourth dimension of gradOutput.</td>
    </tr>
    <tr>
      <td>The last two dimensions of input are 0.</td>
    </tr>  
    <tr>
      <td>The last dimension of grid is not 2.</td>
    </tr>
    <tr>
      <td>The shapes of input and inputGrad are inconsistent, or the shapes of grid and gridGrad are inconsistent.</td>
    </tr> 
  </tbody></table>

## aclnnGridSampler2DBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGridSampler2DBackwardGetWorkspaceSize.</td>
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

- Deterministic computation:
  - **aclnnGridSampler2DBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computing. A deterministic implementation must meet the following conditions:
    - The input and output data types are not DOUBLE.
    - The C axes of the input and output are less than or equal to 2,048.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler2d_backward.h"

namespace {
constexpr uint64_t kNum2 = 2U;
}

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

    // 2. Construct the inputs and outputs based on the API definition.
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    aclBoolArray* outputMask = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
    std::vector<int64_t> inputShape = {1, 1, 5, 8};
    std::vector<int64_t> gridShape = {1, 3, 3, 2};
    std::vector<int64_t> inputGradShape = {1, 1, 5, 8};
    std::vector<int64_t> gridGradShape = {1, 3, 3, 2};
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

    std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> inputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum2);

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
    // Call the first-phase API of aclnnGridSampler2DBackward.
    ret = aclnnGridSampler2DBackwardGetWorkspaceSize(
        gradOutput, input, grid, interpolationMode, paddingMode, alignCorners, outputMask, inputGrad, gridGrad,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGridSampler2DBackward.
    ret = aclnnGridSampler2DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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

    // 7. Release device resources. Modify the code based on the API definition.
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
