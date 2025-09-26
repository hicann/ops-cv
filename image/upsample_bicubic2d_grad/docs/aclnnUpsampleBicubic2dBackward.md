# aclnnUpsampleBicubic2dBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[aclnnUpsampleBicubic2d](../../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)的反向传播。如果输入张量grad_output的shape为(N, C, H, W)，则输出张量grad_input的shape为(N, C, inputSize[2], inputSize[3])。

- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$gradInput(N, C, h, w)$可以表示为：
  
  $$
  {gradInput(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$
  
  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$
  
  其中：
  - $f(h_i, w_j)$是gradOutput在$(h_i, w_j)$的像素值。
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：
    
    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$
    
    其中：
    - $a=-0.75$
    - $d = |(h, w) - (h_i, w_j)|$

## 函数原型

算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnUpsampleBicubic2dBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleBicubic2dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(
  const aclTensor*   gradOut,
  const aclIntArray* outputSize,
  const aclIntArray* inputSize,
  const bool         alignCorners,
  double             scalesH,
  double             scalesW,
  aclTensor*         gradInput,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnUpsampleBicubic2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleBicubic2dBackwardGetWorkspaceSize

- **参数说明**：

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
      <td>gradOut</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式描述中的`gradOutput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与`gradInput`一致。</li><li>当数据格式为ND时，默认按照NCHW格式处理。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCHW、ND、NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>表示输入`gradOut`在H和W维度上的空间大小。对应公式中的`outputSize`。</td>
      <td>size大小为2，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>输入</td>
      <td>表示输出`gradInput`分别在N、C、H、W或N、H、W、C维度上的空间大小。对应公式中的`inputSize`。</td>
      <td>size大小为4，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners</td>
      <td>输入</td>
      <td>决定是否对齐角像素点，对应公式中的`alignCorners`。</td>
      <td>如果为True，则输入和输出张量的角像素点会被对齐，否则不对齐。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH</td>
      <td>输入</td>
      <td>表示输出`gradInput`的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td><!--是否有不能传入负值的约束？没有-->
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW</td>
      <td>输入</td>
      <td>表示输出`gradInput`的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td><!--是否有不能传入负值的约束？没有-->
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与`gradOut`一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCHW、ND、NHWC</td>
      <td>4</td>
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

  - <term>Atlas 训练系列产品</term>：
    - 参数`gradOut`、`gradInput`的数据类型不支持BFLOAT16。
    - 参数`gradOut`、`gradInput`的数据格式不支持NHWC。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  
    参数`gradOut`、`gradInput`的数据格式不支持NHWC。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
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
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOut和gradInput的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOut的维度不为4维。</td>
    </tr>
    </tr>
    <tr>
      <td>outputSize的size大小不等于2。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值小于1。</td>
    </tr>
    </tr>
    <tr>
      <td>inputSize的size大小不等于4。</td>
    </tr>
    <tr>
      <td>inputSize的某个元素值小于1。</td>
    </tr>
    </tr>
    <tr>
      <td>gradOut与inputSize在N、C维度上的size大小不同。</td>
    </tr>
    <tr>
      <td>gradOut在H、W维度上的size大小与outputSize[0]和outputSize[1]未完全相同。</td>
    </tr>
    </tr>
    <tr>
      <td>gradOut和gradInput的N/C轴的维度大小不相等。</td>
    </tr>
    <tr>
      <td>gradOut和gradInput的数据格式不在支持的范围之内。</td>
    </tr>
  </tbody></table>  

## aclnnUpsampleBicubic2dBackward

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBicubic2dBackwardGetWorkspaceSize获取。</td>
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

- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

参数outputSize的H轴和W轴与参数scalesH和参数scalesW，在使用时二选一，即：
  - 当alignCorners为True时：
    - outputSize对应轴的值等于1，scales对应轴的值为0。
    - 其他情况下使用入参inputSize和outputSize中对应轴的参数值，且：$scales=(inputSize-1)/(outputSize-1)$。  
  - 当alignCorners为False时：
    - 当入参scalesH或入参scalesW的值小于等于0时，使用入参outputSize中对应轴的参数值，即：$scales=(inputSize/outputSize)$。
    - 当入参scalesH或入参scalesW的值大于0时，使用入参scalesH或入参scalesW的参数值，即outputSize对应轴的值为$floor(inputSize\_H * scalesH)$，或者$floor(inputSize\_W *   scalesW)$。

<!--确定性计算
- 确定性计算：
  
  aclnnUpsampleBicubic2dBackward默认为非确定性实现，可通过[确定性计算](../../../docs/context/确定性计算.md)配置为确定性实现。确定性实现需同时满足如下条件：
  - inputSize[3] > 130000
  - scaleH >=50
  - scaleW >=50 && inputSize[0] \* inputSize[1] \* inputSize[2] > inputSize[3] \* 0.5
  - scaleH < 0.02 && scaleW < 0.02 && inputSize[0] \* inputSize[1] \* inputSize[2] > inputSize[3] \* 10000
-->
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bicubic_2d_backward.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> outShape = {1, 1, 3, 3};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4.1};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> outArraySize = {2, 2};
  const aclIntArray *outputSize = aclCreateIntArray(outArraySize.data(), outArraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> inputArraySize = {1, 1, 3, 3};
  const aclIntArray *inputSize = aclCreateIntArray(inputArraySize.data(), inputArraySize.size());
  CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUpsampleBicubic2dBackward第一段接口
  ret = aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(self, outputSize, inputSize, 1, 1.1, 1.1, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnUpsampleBicubic2dBackward第二段接口
  ret = aclnnUpsampleBicubic2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dBackward failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(inputSize);

  // 7. 释放device资源，需要根据具体API的接口定义修改
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