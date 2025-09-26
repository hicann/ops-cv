# aclnnUpsampleBilinear2dBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[aclnnUpsampleBilinear2d](../../upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。
- 计算公式：
  - 正向的核心算法逻辑：
    1. 将目标图像缩放到和原始图像一样大的尺寸。
    2. 计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时有不同。则有以下公式：

    $$
    scaleH =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scalesH & alignCorners=false\&scalesH>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scaleW =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scalesW & alignCorners=false\&scalesW>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$
   
    那么，对于output的某个方向上面的点p(x,y)，映射回原始图像中的点记为q(x',y')，则有关系: 
    
    $$
    x' =\begin{cases}
    x * scaleH & alignCorners=true \\
    MAX(0,{(x+0.5)*scaleH-0.5}) & alignCorners=false
    \end{cases}
    $$
    
    $$
    y' =\begin{cases}
    y * scaleW & alignCorners=true \\
    MAX(0,{(y+0.5)*scaleW-0.5}) & alignCorners=false
    \end{cases}
    $$
    
    - 记：
    
      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$
   
    - 则有以下公式：
      $$
      {V(p_{x, y})} = {V(p_{x0, y0})} * {lambda_{0}} * {lambdb_{0}} + {V(p_{x0, y1})} * {lambda_{0}} * {lambdb_{1}} + {V(p_{x1, y0})} * {lambda_{1}} * {lambdb_{0}} + {V(p_{x1, y1})} * {lambda_{1}} * {lambdb_{1}}
      $$

    - 假设：正向插值的输出图像out $(x, y)$受原图像input $(x_i, y_j)$影响，则有:
  
      $$
      gradInput(x_i,y_j) += gradOutput(x,y) * lambd(x_i,y_j)
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnUpsampleBilinear2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBilinear2dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(
  const   aclTensor *gradOut,
  const aclIntArray *outputSize,
  const aclIntArray *inputSize,
  bool               alignCorners,
  double             scalesH,
  double             scalesW,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleBilinear2dBackward(
  void*          workspace,
  uint64_t       workspace_size,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUpsampleBilinear2dBackwardGetWorkspaceSize

- **参数说明**

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
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOutput`。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCHW、NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>表示输入`gradOutput`在H和W维度上的空间大小。对应公式中的`outputSize`。</td>
      <td>size大小为2，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>输入</td>
      <td>表示输出`out`分别在N、C、H和W维度上的空间大小。对应公式中的`inputSize`。</td>
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
      <td>如果设置为True，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值；如果设置为False，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于外界边值，使此操作在保持不变时独立于输入大小scalesH和scalesW。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH</td>
      <td>输入</td>
      <td>表示输出`out`的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td><!--有取值约束不？例如：值为正数才生效。-->
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW</td>
      <td>输入</td>
      <td>表示输出`out`的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据格式需要与入参`gradOut`的数据格式保持一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCHW、NHWC</td>
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

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 参数`gradOut`、`out`的数据类型不支持BFLOAT16.
    - 参数`out`的数据类型需要与`gradOut`的数据类型一致。

- **返回值**

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
      <td>传入的gradOut、outputSize、inputSize或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOut和out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOut的维度不为4维。</td>
    </tr>
    <tr>
      <td>outputSize的size大小不等于2。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值小于1。</td>
    </tr>
    <tr>
      <td>inputSize的size大小不等于4。</td>
    </tr>
    <tr>
      <td>inputSize的某个元素值小于1。</td>
    </tr>
    <tr>
      <td>gradOut与inputSize在N、C维度上的size大小不同。</td>
    </tr>
    <tr>
      <td>gradOut在H、W维度上的size大小与outputSize[0]和outputSize[1]未完全相同。</td>
    </tr>
    <tr>
      <td>gradOut和out的N/C轴的维度大小不相等。</td>
    </tr>
    <tr>
      <td>gradOut和out的数据格式不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleBilinear2dBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBilinear2dBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bilinear_2d_backward.h"

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
  std::vector<int64_t> selfShape = {1, 1, 6, 6};
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

  std::vector<int64_t> outArraySize = {6, 6};
  const aclIntArray *outputSize = aclCreateIntArray(outArraySize.data(), outArraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> inputArraySize = {1, 1, 3, 3};
  const aclIntArray *inputSize = aclCreateIntArray(inputArraySize.data(), inputArraySize.size());
  CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUpsampleBilinear2dBackward第一段接口
  ret = aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(self, outputSize, inputSize, 1, 2, 2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBilinear2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnUpsampleBilinear2dBackward第二段接口
  ret = aclnnUpsampleBilinear2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBilinear2dBackward failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
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