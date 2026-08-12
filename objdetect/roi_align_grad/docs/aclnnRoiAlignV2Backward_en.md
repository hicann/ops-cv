# aclnnRoiAlignV2Backward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_align_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Performs backpropagation of [aclnnRoiAlignV2](../../roi_align/docs/aclnnRoiAlignV2_en.md). RoiAlign is a pooling layer used to input a feature map with a non-uniform size and output a fixed-size feature map.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRoiAlignV2BackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRoiAlignV2Backward** is called to perform computation.

```Cpp
aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(
  const aclTensor*        gradOutput, 
  const aclTensor*        boxes, 
  const aclIntArray*      inputShape, 
  int64_t                 pooledHeight, 
  int64_t                 pooledWidth, 
  float                   spatialScale, 
  int64_t                 samplingRatio, 
  bool                    aligned, 
  aclTensor*              gradInput, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlignV2Backward(
  void*                   workspace, 
  uint64_t                workspaceSize, 
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnRoiAlignV2BackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 298px">
  <col style="width: 184px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 300px">
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
      <th class="tg-0pky">Non-consecutive tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input for backpropagation.</td>
      <td class="tg-0pky">The data type must be the same as that of boxes and gradInput.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4D: (K, C, pooledHeight, pooledWidth)<br>The input gradient tensor for backpropagation has K elements in a batch, and each element has C feature maps with the size of pooledHeight x pooledWidth.<br>K must be consistent with the 0th dimension of boxes.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">boxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coordinates of the region of interest (ROI) box.</td>
      <td class="tg-0pky">The data type must be the same as that of gradOutput and gradInput.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2D, shape: (K, 5)<br>5 indicates the box information (image_id, x1, y1, x2, y2).<br>The value of image_id is in the range of [0, B), which is rounded down to the image ID. B is the first value of inputShape.<br>The coordinates must meet these requirements: 0 ≤ x1 ≤ x2 ≤ inputWidth/spatialScale and 0 ≤ y1 ≤ y2 ≤ inputHeight/spatialScale.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">inputShape (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Shape of the forward input, which is used to specify the output shape of backpropagation.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32, INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">The size is 4 and the value is (B, C, inputHeight, inputWidth).<br>It indicates that the forward RoiAlign input tensor contains B images in a batch, and each image has C feature maps with the size of inputHeight x inputWidth.</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledHeight (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Height of the output image after forward RoiAlign pooling.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledWidth (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Width of the output image after forward RoiAlign pooling.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">spatialScale (float)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Multiplicative spatial scale factor, which is used to convert the ROI coordinates from the input spatial scale to the scale used for pooling, that is, the spatial scale of the input feature map X relative to the input image.</td>
      <td class="tg-0pky">The value must be greater than 0.</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">samplingRatio (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Sampling frequency used to calculate each output element in the H and W directions in RoiAlign.</td>
      <td class="tg-0pky">Must be greater than or equal to 0.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">aligned (bool)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">If the value is false, the implementation of the <a href="./aclnnRoiAlign.md">aclnnRoiAlign</a> version is aligned.<br>If the value is true, the pixel offset of the box coordinate is -0.5 to better align the adjacent pixel indexes.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output of backpropagation.</td>
      <td class="tg-0pky">The data type must be the same as that of gradOutput and boxes.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4D, with shape (B, C, inputHeight, inputWidth)</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input gradOutput, boxes, inputShape, and gradInput are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of gradOutput and gradInput are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, boxes, inputShape, and gradInput do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The value of spatialScale must be greater than 0, and the value of samplingRatio must be greater than or equal to 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlignV2Backward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRoiAlignV2BackwardGetWorkspaceSize.</td>
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
  - **aclnnRoiAlignV2Backward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computing.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2_backward.h"

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

template <typename T>
int CreateAclNchTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
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
  std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
  std::vector<int64_t> boxesShape = {1, 5};
  std::vector<int64_t> inputShape = {1, 1, 6, 6};

  void* gradOutputDeviceAddr = nullptr;
  void* boxesDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* boxes = nullptr;
  aclTensor* gradInput = nullptr;

  std::vector<float> gradOutputHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};
  std::vector<float> boxesHostData = {0.0, -2.0, -2.0, 22.0, 22.0};
  std::vector<float> gradInputHostData = {1.125, 1.125, 1.625, 1.625, 2.125, 2.125, 1.125, 1.125, 1.625, 1.625, 2.125, 2.125,
                                    4.125, 4.125, 4.625, 4.625, 5.125, 5.125, 4.125, 4.125, 4.625, 4.625, 5.125, 5.125,
                                    7.125, 7.125, 7.625, 7.625, 8.125, 8.125, 7.125, 7.125, 7.625, 7.625, 8.125, 8.125};

  // Create a gradOutput aclTensor.
  ret = CreateAclNchTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create boxes aclTensor.
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an inputShape aclIntArray.
  const aclIntArray *inputShapeArray = aclCreateIntArray(inputShape.data(), inputShape.size());
  CHECK_RET(inputShapeArray != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  // Create a gradInput aclTensor.
  ret = CreateAclNchTensor(gradInputHostData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int64_t pooledHeight = 3;
  int64_t pooledWidth = 3;
  int64_t samplingRatio = 2;
  float spatialScale = 0.25f;
  bool aligned = false;

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoiAlignV2Backward.
  ret = aclnnRoiAlignV2BackwardGetWorkspaceSize(gradOutput, boxes, inputShapeArray, pooledHeight, pooledWidth, spatialScale, 
                                              samplingRatio, aligned, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2BackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRoiAlignV2Backward.
  ret = aclnnRoiAlignV2Backward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2Backward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(inputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    gradInputDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(boxes);
  aclDestroyIntArray(inputShapeArray);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(boxesDeviceAddr);
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
