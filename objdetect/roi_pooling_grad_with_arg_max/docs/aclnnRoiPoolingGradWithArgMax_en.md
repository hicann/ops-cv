# aclnnRoiPoolingGradWithArgMax

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |    ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs backpropagation of RoiPoolingWithArgMax. Traverse the pooling result of each ROI and accumulate the backward gradient contributions on the coordinates of the feature map, that is, complete the backward propagation on the entire image.
- Formulas:

  $$
  \frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}
  $$
  
  Wherein:
  
  $$
  [i = i^*(r,j)]  = \begin{cases} 1, & i^*(r,j) \geq 1 \\ 0, & otherwise \end{cases}
  $$

  The decision function [i = i^*(r,j)] indicates whether the node i is selected as the maximum value output by the jth output node of the candidate region r.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnRoiPoolingGradWithArgMax` is called to perform computation.

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(
    const aclTensor*      gradOutput,
    const aclTensor*      gradInputRef,
    const aclTensor*      rois,
    const aclTensor*      argmax,
    int64_t               pooledH,
    int64_t               pooledW,
    double                spatialScale,
    uint64_t*             workspaceSize,
    aclOpExecutor**       executor);
```

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMax(
  void*                   workspace, 
  uint64_t                workspace_size, 
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup> 
    <col style="width: 200px"> 
    <col style="width: 120px"> 
    <col style="width: 250px"> 
    <col style="width: 120px"> 
    <col style="width: 212px"> 
    <col style="width: 120px">  
    <col style="width: 250px">  
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
      <td>Gradient input.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4D, with shape of (roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInputRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Output result.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4D, with shape of (N, C, H, W)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>rois (aclTensor*)</td>
      <td>Input</td>
      <td>ROI area.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2D, with shape of (roisN, 5).<br>5: (batchId, x1, x2, y1, y2)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>argmax (aclTensor*)</td>
      <td>Input</td>
      <td>Index of the specified target gradient.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>4D, with shape (roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pooledH (int64_t)</td>
      <td>Attribute</td>
      <td>Pooling height.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>pooledW (int64_t)</td>
      <td>Attribute</td>
      <td>Pooling width.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatialScale (double)</td>
      <td>Attribute</td>
      <td>Scale factor for mapping the input coordinates to the ROI coordinates.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>The input gradOutput, rois, argmax, and gradInputRef are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data types of gradOutput, rois, argmax, and gradInputRef are not supported.</td>
    </tr>
    <tr>
      <td>gradOutput, argmax, and gradInputRef have the same data type.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput, argmax, and gradInputRef is 4, and the shape of rois is 2.</td>
    </tr>
    <tr>
      <td>The shape[0] of gradOutput, argmax, and rois are the same.</td>
    </tr>
    <tr>
      <td>The shape[1] of gradOutput and argmax are the same.</td>
    </tr>
    <tr>
      <td>The shape[2] of gradOutput and argmax is equal to pooledH, and the shape[3] is equal to pooledW.</td>
    </tr>
    <tr>
      <td>The value of rois is greater than or equal to 0.</td>
    </tr>
    <tr>
      <td>The values of pooledH and pooledW are greater than 0.</td>
    </tr>
    <tr>
      <td>rois[:, 1] is less than rois[:, 2] and rois[:, 3] is less than rois[:, 4].</td>
    </tr>
    <tr>
      <td>The values of rois.shape[0] and gradOutput.shape[0] are less than or equal to 1024.</td>
    </tr>
    <tr>
      <td>The value of gradInputRef.shape[1] is equal to gradOutput.shape[1].</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiPoolingGradWithArgMax

* **Parameters**:

  <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
      <col style="width: 150px">
      <col style="width: 100px">
      <col style="width: 650px">
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
          <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRoiPoolingGradWithArgMax.</td>
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
      </tbody></table>
      </div>

* **Return values:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. The data types of gradOutput, rois, argmax, and gradInputRef are within the supported range.
2. gradOutput, argmax, and gradInputRef have the same data type.
3. The shape of gradOutput, argmax, and gradInputRef is 4, and the shape of rois is 2.
4. The shape[0] of gradOutput, argmax, and rois is the same.
5. The shape[1] of gradOutput and argmax is the same.
6. The shape[2] of gradOutput and argmax is equal to pooledH and shape[3] is equal to pooledW.
7. The value of rois is greater than or equal to 0.
8. pooledH and pooledW are greater than 0.
9. rois[:, 1] is less than rois[:, 2] and rois[:, 3] is less than rois[:, 4].
10. rois.shape[0] and gradOutput.shape[0] are less than or equal to 1024.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_grad_with_arg_max.h"
#include <iostream>
using namespace std;


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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // (Boilerplate) Perform initialization.
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
    // 2. Allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. Call aclrtMemcpy to copy the data from the host to the device memory.
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
    // 1. Call ACL to initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    aclTensor* gradOutput = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 32, 2, 2};
    std::vector<float> gradOutputHostData(128, 1.0); // 2048: Create a vector containing 2048 elements (32 x 4 x 4 x 4).
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    std::vector<int64_t> xShape = {4, 3, 3, 32};
    std::vector<float> xHostData(1152, 1.0);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* rois = nullptr;
    void* roisDeviceAddr = nullptr;
    std::vector<int64_t> roisShape = {1, 5};
    std::vector<float> roisHostData = {0.0, 0.0, 1.0, 0.0, 1.0};
    ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* argmax = nullptr;
    void* argmaxDeviceAddr = nullptr;
    std::vector<int64_t> argmaxShape = {1, 32, 2, 2};
    std::vector<int32_t> argmaxHostData(128, 3.0);
    ret = CreateAclTensor(argmaxHostData, argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int32_t pooledH = 2;
    int32_t pooledW = 2;
    double spatialScale = 1.0;

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {4, 32, 3, 3};
    std::vector<float> outHostData(1152, 0.0);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. Call the first part of the aclnnAddExample API.
    ret = aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(gradOutput, x, rois, argmax, pooledH, pooledW, spatialScale, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. Call the second part of the aclnnRoiPoolingGradWithArgMax API.
    ret = aclnnRoiPoolingGradWithArgMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMax failed. ERROR: %d\n", ret); return ret);

    // 6. (Fixed) Synchronize and wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnRoiPoolingGradWithArgMax run success.\n");

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(outShape, &outDeviceAddr);

    // 7. Destroy the aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(x);
    aclDestroyTensor(rois);
    aclDestroyTensor(argmax);
    aclDestroyTensor(out);

    // 8. Release device resources.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(roisDeviceAddr);
    aclrtFree(argmaxDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 9. Deinitialize ACL.
    aclFinalize();

    return 0;
}

```
