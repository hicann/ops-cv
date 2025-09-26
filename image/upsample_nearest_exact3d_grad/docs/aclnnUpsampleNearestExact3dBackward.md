# aclnnUpsampleNearestExact3dBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[aclnnUpsampleNearestExact3d](../../upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md)的反向计算。
- 计算公式：
  
  $$
  scalesD = outputSize[0] / inputSize[2]
  $$
  $$
  scalesH = outputSize[1] / inputSize[3]
  $$
  $$
  scalesW = outputSize[2] / inputSize[4]
  $$
  $$
  gradInput(N, C, floor ( scalesD * ( D + 0.5 )), floor ( scalesH * ( H + 0.5 )),  floor ( scalesW * ( W+ 0.5 ))) += gradOutput( N, C, D, H ,W)
  $$

## 函数原型
每个算子分为[两段式接口](./../../../docs/context/两段式接口.md)，必须先调用“aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleNearestExact3dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut, 
  const aclIntArray *outputSize, 
  const aclIntArray *inputSize, 
  double             scalesD, 
  double             scalesH, 
  double             scalesW, 
  aclTensor         *gradInput, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```
```Cpp
aclnnStatus aclnnUpsampleNearestExact3dBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 392px">
  <col style="width: 228px">
  <col style="width: 101px">
  <col style="width: 100px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示反向计算的的梯度Tensor，对应公式中的`gradOutput`。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>表示输入`gradOut`在D、H和W维度上的空间大小，对应公式中的`outputSize`。</td>
      <td>size大小为3，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>输入</td>
      <td>表示输出`gradInput`分别在N、C、D、H和W维度上的空间大小，对应公式中的`inputSize`。</td>
      <td>size大小为5，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD</td>
      <td>输入</td>
      <td>表示输出`gradInput`的depth维度乘数，对应公式中的`scalesD`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH</td>
      <td>输入</td>
      <td>表示输出`gradInput`的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW</td>
      <td>输入</td>
      <td>表示输出`gradInput`的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的输出`gradInput`。</td>
      <td><ul><li>不支持空tensor。</li><li>数据类型和数据格式与入参`gradOut`的数据类型和数据格式保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOut、outputSize、inputSize或gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>gradOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOut和gradInput的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOut的维度不为5维。</td>
    </tr>
    <tr>
      <td>outputSize的size大小不等于3。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>inputSize的size大小不等于5。</td>
    </tr>
    <tr>
      <td>gradOut与inputSize在N、C维度上的size大小不同。</td>
    </tr>
    <tr>
      <td>gradOut在D、H、W维度上的size大小与outputSize[0]、outputSize[1]和outputSize[2]未完全相同。</td>
    </tr>
    <tr>
      <td>gradInput在N、C维度的size大小与inputSize[0]、inputSize[1]未完全相同。</td>
    </tr>
    <tr>
      <td>gradInput在D、H、W维度的size大小与inputSize[2]、inputSize[3]、inputSize[4]未完全相同。</td>
    </function>
  </body>
</html>
  </tbody></table>

## aclnnUpsampleNearestExact3dBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

## 约束说明

- 输入数据缩放场景放大倍数必须小于等于50，即$outputSize[0]/输出shape的高度H$、$outputSize[1]/输出shape的宽度W$以及$outputSize[2]/输出shape的深度D$必须小于等于50。
- 参数outputSize与参数scalesD、scalesH、scalesW，在使用时二选一，即：
  - 当入参scalesD、scalesH、scalesW，其中一个参数的值小于等于0时，使用入参outputSize的参数值。
  - 当入参scalesD、scalesH、scalesW的值都大于0时，使用入参scalesD、scalesH、scalesW的参数值，且$outputSize=[floor(inputSize\_D * scalesD)，floor(inputSize\_H * scalesH)，floor(inputSize\_W * scalesW)]$。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_exact3d_backward.h"

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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradOutShape = {2, 2, 2, 2, 2};
  std::vector<int64_t> gradInputShape = {2, 2, 1, 1, 1};
  void* gradOutDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                        10, 11, 12, 13, 14, 15, 16, 17, 18,
                                        19, 20, 21, 22, 23, 24, 25, 26, 27,
                                        28, 29, 30, 31, 32};
  std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
  std::vector<int64_t> outputSizeData = {2, 2, 2};
  std::vector<int64_t> inputSizeData = {2, 2, 1, 1, 1};
  double scalesD = 0.0;
  double scalesH = 0.0;
  double scalesW = 0.0;
  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const aclIntArray *outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  const aclIntArray *inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
  CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUpsampleNearestExact3dBackward第一段接口
  ret = aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(gradOut, outputSize, inputSize, scalesD, scalesH, scalesW, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnUpsampleNearestExact3dBackward第二段接口
  ret = aclnnUpsampleNearestExact3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearestExact3dBackward failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源，需要根据具体API的接口定义修改
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
