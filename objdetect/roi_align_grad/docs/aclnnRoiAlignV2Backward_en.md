# aclnnRoiAlignV2Backward

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_align_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Operator description: Performs backpropagation of [aclnnRoiAlignV2](../../roi_align/docs/aclnnRoiAlignV2_en.md). RoiAlign is a pooling layer used to input a feature map with a non-uniform size and output a fixed-size feature map.

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRoiAlignV2BackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRoiAlignV2Backward** is called to perform computation.

- `aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* boxes, const aclIntArray* inputShape, int64_t pooledHeight, int64_t pooledWidth, float spatialScale, int64_t samplingRatio, bool aligned, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnRoiAlignV2Backward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRoiAlignV2BackwardGetWorkspaceSize

- **Parameters:**

  - **gradOutput** (aclTensor\*, compute input): input of backpropagation. aclTensor on the device. The data type can be FLOAT, which must be the same as those of **boxes** and **gradInput**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW. The shape is 4D (K, C, pooledHeight, pooledWidth), indicating that the input gradient tensor for backpropagation has **K** elements in a batch, and each element has **C** feature maps whose size is pooledHeight \* pooledWidth. **K** must be the same as the 0th dimension of **boxes**.

  - **boxes** (aclTensor\*, compute input): box coordinates of the ROI. aclTensor on the device. The data type can be FLOAT, which must be the same as that of **gradOutput** and **gradInput**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape is 2D (K, 5), where **5** indicates the box information (image_id, x1, y1, x2, y2). The value range of image_id is \[0, B\), rounded down to an integer multiple of the image ID. **B** is the first value of **inputShape**. The coordinates must meet these requirements: 0 ≤ x1 ≤ x2 ≤ inputWidth/spatialScale and 0 ≤ y1 ≤ y2 ≤ inputHeight/spatialScale.

  - **inputShape** (aclIntArray\*, compute input): **shape** of the forward input, which is used to specify the **shape** of the backpropagation output. aclIntArray on the host. The supported data types are INT32 and INT64. The size is **4**, and the shape is (B, C, inputHeight, inputWidth), indicating that the forward RoiAlign input tensor has **B** images in a batch, and each image has **C** feature maps whose size is inputHeight \* inputWidth.

  - **pooledHeight** (int64_t, compute input): height of the output image after forward RoiAlign pooling. It is an input parameter on the host.

  - **pooledWidth** (int64_t, compute input): width of the output image after forward RoiAlign pooling. It is an input parameter on the host.
  
  - **spatialScale** (float, compute input): multiplication spatial scale factor. It is used to convert the ROI coordinates from the input spatial scale to the scale used for pooling, that is, the spatial scale of the input feature map X relative to the input image. It is an input parameter on the host. The value must be greater than 0.

  - **samplingRatio** (int64_t, compute input): sampling frequency for calculating each output element in the H and W directions in RoiAlign. It is an input parameter on the host. The value must be greater than or equal to 0.
  
  - **aligned** (bool, compute input): If the value is false, [aclnnRoiAlign](../../roi_align/docs/aclnnRoiAlign_en.md) is implemented. If the value is true, the box coordinates are pixel shifted by -0.5 to better align adjacent pixel indexes. It is an input parameter on the host.

  - **gradInput** (aclTensor\*, compute output): output of backpropagation. aclTensor on the device. The data type can be FLOAT, which must be the same as that of **gradOutput** and **boxes**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW. The shape is 4D (B, C, inputHeight, inputWidth).

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): The input gradOutput, boxes, inputShape, and gradInput are null pointers.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types and data formats of gradOutput and gradInput are not supported.
                                  2. The shapes of gradOutput, boxes, inputShape, and gradInput do not meet the restrictions.
                                  3. spatialScale must be greater than 0, and samplingRatio must be greater than or equal to 0.
```

## aclnnRoiAlignV2Backward

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnRoiAlignV2BackwardGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

## Constraints

- Deterministic computing:
  - **aclnnRoiAlignV2Backward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computing.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_cv.md).

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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRoiAlignV2Backward.
  ret = aclnnRoiAlignV2Backward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2Backward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(inputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    gradInputDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(boxes);
  aclDestroyIntArray(inputShapeArray);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
