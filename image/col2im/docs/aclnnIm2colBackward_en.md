# aclnnIm2colBackward

[📄 View Source Code](https://gitcode.com/cann/ops-cv/tree/master/image/col2im)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Operator description: Extracts sliding local blocks from a batched input tensor and combines an array of sliding local blocks into a large tensor.
- Formulas:
  
  Consider a batched input tensor with shape $(N,C,*)$, where $N$ is the batch dimension, $C$ is the channel dimension, and $*$ represents any spatial dimension.

  This operation flattens each sliding block of size kernel_size in the input spatial dimension into a column (i .e., the last dimension) of a 3D output tensor with shape $(N,C×\prod(kernel\_size),L)$.

  The values are as follows:
  - $C×\prod(kernel\_size)$ is the number of values in each block (a block has $\prod(kernel\_size)$ spatial locations, and each spatial location contains a $C$ channel vector), and $L$ is the total number of these blocks.

    $$
    L=\prod_d⌊{\frac{spatial\_size[d]+2×padding[d]−dilation[d]×(kernel\_size[d]−1)−1}{stride[d]}+1}⌋
    $$

  - The spatial_size is formed by the spatial dimensions of the input (the $*$ above), and $d$ covers all spatial dimensions.
  Therefore, in the last dimension (column) index, the **output** gives all values in a block.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnIm2colBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnIm2colBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput, 
  const aclIntArray *inputSize, 
  const aclIntArray *kernelSize, 
  const aclIntArray *dilation, 
  const aclIntArray *padding, 
  const aclIntArray *stride, 
  aclTensor         *out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnIm2colBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnIm2colBackwardGetWorkspaceSize

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
  <col style="width: 165px">
  <col style="width: 121px">
  <col style="width: 325px">
  <col style="width: 272px">
  <col style="width: 252px">
  <col style="width: 121px">
  <col style="width: 149px">
  <col style="width: 141px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>shape is (C×kernel_size[0]×kernel_size[1],L) or (N,C×kernel_size[0]×kernel_size[1],L).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2D and 3D are supported.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>Input</td>
      <td>Shape of the input tensor.</td>
      <td>The value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the convolution kernel.</td>
      <td>The value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Dilation parameter.</td>
      <td>The value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Padding size of the convolution.</td>
      <td>The value is greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the convolution.</td>
      <td>The value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The shape is (C, spatial_size[0], spatial_size[1]) or (N, C, spatial_size[0], spatial_size[1]).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>The input can be 3D or 4D, and the dimension is one more than that of gradOutput.</td>
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
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1124px"><colgroup>
  <col style="width: 284px">
  <col style="width: 124px">
  <col style="width: 716px">
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
      <td>The input gradOutput, inputSize, kernelSize, dilation, padding, stride, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of gradOutput is not supported.</td>
    </tr>
    <tr>
      <td>The dimension of gradOutput is neither 2 nor 3.</td>
    </tr>
    <tr>
      <td>When gradOutput is 2D, out is not 3D. When gradOutput is 3D, out is not 4D.</td>
    </tr>
    <tr>
      <td>The size of inputSize, kernelSize, dilation, padding, or stride is not 2.</td>
    </tr>
    <tr>
      <td>The value of kernelSize, dilation, or stride is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The value of padding is less than 0.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput does not comply with the shape specified in the parameter description.</td>
    </tr>
    <tr>
      <td>The shape of out does not comply with the shape specified in the parameter description.</td>
    </tr>
  </tbody>
  </table>

## aclnnIm2colBackward

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 180px">
  <col style="width: 130px">
  <col style="width: 839px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnIm2colBackwardGetWorkspaceSize.</td>
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

None

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_im2col_backward.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> gradShape = {4, 2};
  std::vector<int64_t> outShape = {1, 1, 1};

  void* gradDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclIntArray* inputSize = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* dilation = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* stride = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  std::vector<int64_t> inputSizeData = {1, 1};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> dilationData = {1, 1};
  std::vector<int64_t> paddingData = {1, 1};
  std::vector<int64_t> strideData = {1, 2};
  std::vector<float> outHostData = {0.0};

  // Create a grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an aclIntArray.
  inputSize = aclCreateIntArray(inputSizeData.data(), 2);
  CHECK_RET(inputSize != nullptr, return ret);
  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  CHECK_RET(kernelSize != nullptr, return ret);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  CHECK_RET(dilation != nullptr, return ret);
  padding = aclCreateIntArray(paddingData.data(), 2);
  CHECK_RET(padding != nullptr, return ret);
  stride = aclCreateIntArray(strideData.data(), 2);
  CHECK_RET(stride != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnIm2colBackward.
  ret = aclnnIm2colBackwardGetWorkspaceSize(grad, inputSize, kernelSize, dilation, padding, stride, out,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2colBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnIm2colBackward.
  ret = aclnnIm2colBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2colBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclIntArray. Modify the code based on the API definition.
  aclDestroyTensor(grad);
  aclDestroyIntArray(inputSize);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(dilation);
  aclDestroyIntArray(padding);
  aclDestroyIntArray(stride);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradDeviceAddr);
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
