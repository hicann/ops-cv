# aclnnUpsampleNearest3d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1.将目标图像缩放到和原始图像一样大的尺寸。
    2.对于缩放之后的目标图像的点，计算距离最近的原始图像的点，后者的值直接复制给前者。
  - 具体计算逻辑：
    则计算缩放系数有以下公式：
    
    $$
    scale = self.dim / outputSize
    $$
    
    那么，对于out的某个方向上面的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系: 
    
    $$
    x' = \min(\lfloor x * scaleD \rfloor, self.dim(2) - 1)
    $$

    $$
    y' = \min(\lfloor y * scaleH \rfloor, self.dim(3) - 1)
    $$
    
    $$
    z' = \min(\lfloor z * scaleW \rfloor, self.dim(4) - 1)
    $$
    
    则有以下公式：
    
    $$
    {V(p_{x,y,z})} = {V(q_{x',y',z'})}
    $$
    
## 函数原型
每个算子分为[两段式接口](./../../../docs/context/两段式接口.md)，必须先调用`aclnnUpsampleNearest3dGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnUpsampleNearest3d`接口执行计算。

```cpp
aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(
  const aclTensor   *self, 
  const aclIntArray *outputSize, 
  double             scalesD, 
  double             scalesH, 
  double             scalesW, 
  aclTensor         *out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnUpsampleNearest3d(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  const aclrtStream stream)
```


## aclnnUpsampleNearest3dGetWorkspaceSize

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
      <td>self</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量。对应公式中的`self`。</td>
      <td><ul><li>不支持空tensor。</li><li>当数据格式为ND时，默认按照NCDHW格式处理。</li><li>self的所有轴取值均要满足小于等于(2^31-1)。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>指定输出out的Tensor大小，对应公式中的`outputSize`。</td>
      <td><ul><li>各元素均大于零。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD</td>
      <td>输入</td>
      <td>指定输出out的depth维度乘数，对应公式中的`scalesD`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH</td>
      <td>输入</td>
      <td>指定输出out的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW</td>
      <td>输入</td>
      <td>指定输出out的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>            
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中`out`的点p坐标。</td>
      <td><ul><li>不支持空tensor。</li><li>数据类型和数据格式需与入参self的数据类型和数据格式保持一致。</li><li>out的所有轴取值均要满足小于等于(2^31-1)。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>            
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

    - <term>Atlas 推理系列产品</term>：
    
      入参`self`和出参`out`的数据类型不支持BFLOAT16、DOUBLE。
    - <term>Atlas 训练系列产品</term>：
    
      入参`self`和出参`out`的数据类型不支持BFLOAT16。

  
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>self的数据类型不在支持的范围之内或self与out数据类型不同。</td>
    </tr>
    <tr>
      <td>self的shape不是5维。</td>
    </tr>
    <tr>
      <td>outputSize的size大小不等于3。</td>
    </tr>
    <tr>
      <td>self在D、H、W维度上的size不大于0。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>self的C维度为0。</td>
    </tr>
    <tr>
      <td>out的shape不等于由self和outputSize推导得到shape。</td>
    </tr>                    
  </tbody></table>

## aclnnUpsampleNearest3d

- **参数说明**
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearest3dGetWorkspaceSize获取。</td>
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

参数outputSize与参数scalesD、scalesH、scalesW，在使用时二选一，即：
- 当入参scalesD、scalesH、scalesW，其中一个参数的值小于等于0时，使用入参outputSize的参数值。
- 当入参scalesD、scalesH、scalesW的值都大于0时，使用入参scalesD、scalesH、scalesW的参数值，且$outputSize=[floor(self\_D * scalesD)，floor(self\_H * scalesH)，floor(self\_W * scalesW)]$。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_3d.h"

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

  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  std::vector<int64_t> selfShape = {2, 2, 2, 2, 3};
  std::vector<int64_t> outShape = {2, 2, 1, 1, 1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24, 25, 26, 27,28, 29, 30, 31, 32, 33, 34,
                                     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
  std::vector<float> outHostData = {1, 2, 3, 4};
  std::vector<int64_t> outputSizeData = {1, 1, 1};
  double scalesD = 0.0;
  double scalesH = 0.0;
  double scalesW = 0.0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const aclIntArray *outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUpsampleNearest3d第一段接口
  ret = aclnnUpsampleNearest3dGetWorkspaceSize(self, outputSize, scalesD, scalesH, scalesW, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnUpsampleNearest3d第二段接口
  ret = aclnnUpsampleNearest3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3d failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

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

