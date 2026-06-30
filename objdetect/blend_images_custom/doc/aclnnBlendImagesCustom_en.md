# aclnnBlendImagesCustom

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Operator description: Performs transparency multiplication on tensors **rgb**, **frame**, and **alpha**.

- Formula:

$$
out_{i*3}=rgb_{i*3} * (alpha_i / 255) + frame_{i*3}*(1 - alpha_i/255)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBlendImagesCustomGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnBlendImagesCustom** is called to perform computation.

* `aclnnStatus aclnnBlendImagesCustomGetWorkspaceSize(const aclTensor *rgb, const aclTensor *alpha, const aclTensor *frame, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnBlendImagesCustom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBlendImagesCustomGetWorkspaceSize

- **Parameters:**
  
  - **rgb** (aclTensor*, compute input): aclTensor on the device. The data type can be UINT8, and the shape can be HWC (C=3). It meets the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with **alpha**. Only contiguous tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **alpha** (aclTensor*, compute input): aclTensor on the device. The data type can be UINT8, and the shape can be HWC (C = 1). It meets the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with **rgb**. Only contiguous tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **frame** (aclTensor*, compute input): aclTensor on the device. The data type can be UINT8, and the shape can be HWC (C = 3). It meets the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with **alpha**. Only contiguous tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **out** (aclTensor*, compute output): aclTensor on the device. The data type can be UINT8, and the shape can be HWC (C = 3). It is the same as the shape of **frame**. Only contiguous tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed rgb, alpha, frame, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type and data format of rgb, alpha, and frame are not supported.
                                    2. The shapes of rgb, alpha, and frame cannot be broadcast. rgb and frame support HWC (C = 3), and alpha supports HWC (C = 1).
  ```

## aclnnBlendImagesCustom

- **Parameters:**
  - **workspace** (void \*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBlendImagesCustomGetWorkspaceSize**.
  - **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_cv.md).

## Constraints
- Deterministic computing:
  - **aclnnBlendImagesCustom** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_cv.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_blend_images_custom.h"

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
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> rgbShape = {4, 3};
  std::vector<int64_t> alphaShape = {4, 1};
  std::vector<int64_t> frameShape = {4, 3};
  std::vector<int64_t> outShape = {4, 3};

  void* rgbDeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  void* frameDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* rgb = nullptr;
  aclTensor* alpha = nullptr;
  aclTensor* frame = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> rgbHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<float> alphaHostData = {255, 255, 255, 255};
  std::vector<float> frameHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<float> outHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

  ret = CreateAclTensor(rgbHostData, rgbShape, &rgbDeviceAddr, aclDataType::ACL_UINT8, &rgb);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_UINT8, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(frameHostData, frameShape, &frameDeviceAddr, aclDataType::ACL_UINT8, &frame);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnBlendImagesCustom.
  ret = aclnnBlendImagesCustomGetWorkspaceSize(rgb, alpha, frame, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlendImagesCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnBlendImagesCustom.
  ret = aclnnBlendImagesCustom(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlendImagesCustom failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(rgb);
  aclDestroyTensor(alpha);
  aclDestroyTensor(frame);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(rgbDeviceAddr);
  aclrtFree(alphaDeviceAddr);
  aclrtFree(frameDeviceAddr);
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
