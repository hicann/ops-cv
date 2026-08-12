# aclnnRoiAlign

[📄 View source code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_align)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

RoiAlign is a pooling layer used to input a feature map with a non-uniform size and output a fixed-size feature map.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRoiAlignGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRoiAlign** is called to perform computation.

```Cpp
aclnnStatus aclnnRoiAlignGetWorkspaceSize(
  const aclTensor*        self, 
  const aclTensor*        rois, 
  const aclTensor*        batchIndices, 
  const char*             mode, 
  int                     outputHeight, 
  int                     outputWidth, 
  int                     samplingRatio, 
  float                   spatialScale, 
  aclTensor*              out, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlign(
  void*                   workspace, 
  uint64_t                workspaceSize, 
  aclOpExecutor*          executor, 
  const aclrtStream       stream)
```

## aclnnRoiAlignGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 330px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 219px">
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
      <th class="tg-0pky">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">Must be the same as the data type of the rois/out.</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4D, which is (N, C, H, W)</td>.
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">rois (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Region of interest.</td>
      <td class="tg-0pky"><ul><li>Must be the same as the data type of the self/out. </li><li>The coordinate format is (x1, y1, x2, y2), and 0 <= x1 <= x2 <= W/spatialScale and 0 <= y1 <= y2 <= H/spatialScale must be met.</li></ul></td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2D, which is (numRois, 4)</td>.
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">batchIndices (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Index of the image corresponding to each batch.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1D, which is (numRois,)</td>.
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mode (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">The value can be "avg" or "max". It indicates the pooling mode.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">String</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">outputHeight (int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Height of the output image.</td>
      <td class="tg-0lax">The recommended value is 1.</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">outputWidth (int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Width of the output image.</td>
      <td class="tg-0lax">The recommended value is 1.</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">samplingRatio (int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Sampling frequency for each output element in the H and W directions.</td>
      <td class="tg-0lax">The recommended value is 0.</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">spatialScale (float)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Multiplicative spatial scale factor, which converts the ROI coordinates from the input spatial scale to the scale used for pooling, that is, the spatial scale of the input feature map X relative to the input image.</td>
      <td class="tg-0lax">The recommended value is 1.0.</td>
      <td class="tg-0lax">FLOAT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output tensor.</td>
      <td class="tg-0lax">Must be consistent with the data type of self/rois.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">NCHW</td>
      <td class="tg-0lax">4D, which is (numRois, C, outputHeight, outputWidth)</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator computation process.</td>
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
      <td>The input self, rois, batchIndices, and out are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>self, rois, and out support only FLOAT and FLOAT16.</td>
    </tr>
    <tr>
      <td>batchIndices supports only INT32.</td>
    </tr>
    <tr>
      <td>The data types of self, rois, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>self and out support NCHW.</td>
    </tr>
    <tr>
      <td>rois and batchIndices support ND.</td>
    </tr>
    <tr>
      <td>self and out must be 4-dimensional.</td>
    </tr>
    <tr>
      <td>rois must be 2-dimensional.</td>
    </tr>
    <tr>
      <td>batchIndices must be 1-dimensional.</td>
    </tr>
    <tr>
      <td>mode can only be avg or max.</td>
    </tr>
    <tr>
      <td>samplingRatio must be greater than or equal to 0.</td>
    </tr>
    <tr>
      <td>spatialScale must be greater than 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlign

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRoiAlignGetWorkspaceSize.</td>
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
  - **aclnnRoiAlign** defaults to a deterministic implementation.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align.h"

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
  std::vector<int64_t> selfShape = {1, 1, 6, 6};
  std::vector<int64_t> roisShape = {1, 4};
  std::vector<int64_t> batchIndicesShape = {1};
  std::vector<int64_t> outShape = {1, 1, 3, 3};

  void* selfDeviceAddr = nullptr;
  void* roisDeviceAddr = nullptr;
  void* batchIndicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* rois = nullptr;
  aclTensor* batchIndices = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<float> roisHostData = {-2.0, -2.0, 22.0, 22.0};
  std::vector<int32_t> batchIndicesHostData = {0};
  std::vector<float> outHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};

  // Create a self aclTensor.
  ret = CreateAclNchTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a rois aclTensor.
  ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a batchIndices aclTensor.
  ret = CreateAclTensor(batchIndicesHostData, batchIndicesShape, &batchIndicesDeviceAddr, aclDataType::ACL_INT32, &batchIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclNchTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  const char* mode = "avg";
  int outputHeight = 3;
  int outputWidth = 3;
  int samplingRatio = 0;
  float spatialScale = 1.0f;

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoiAlign.
  ret = aclnnRoiAlignGetWorkspaceSize(self, rois, batchIndices, mode, outputHeight, outputWidth, samplingRatio, spatialScale, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRoiAlign.
  ret = aclnnRoiAlign(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlign failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(rois);
  aclDestroyTensor(batchIndices);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(roisDeviceAddr);
  aclrtFree(batchIndicesDeviceAddr);
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
