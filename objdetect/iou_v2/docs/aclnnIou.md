# aclnnIou

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/iou_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：对两个输入矩形框集合，计算交并比（IOU）或前景交叉比（IOF），用于评价预测框（bBox）和真值框（gtBox）的重叠度。
- 计算公式：

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2} 
  $$
  其中，Area_1为bBox的面积，Area_2为gtBox的面积，Area_3为两者重叠部分面积，x和y的定义见参数说明。
  $$
  Area_1 = (X_1 - X_0)(Y_1 - Y_0) \\
  Area_2 = (X_3 - X_2)(Y_3 - Y_2) \\
  Area_3 = max( min(X_1, X_3) - max(X_0, X_2), 0 ) * max( min(Y_1, Y_3) - max(Y_0, Y_2), 0 )
  $$

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnIouGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIou”接口执行计算。

- `aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor* bBoxes, const aclTensor* gtBoxes, const char* mode, float eps, bool aligned, aclTensor* overlap, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnIou(void* workspace, uint64_t workspaceSize,  aclOpExecutor* executor, aclrtStream stream)`

## aclnnIouGetWorkspaceSize

- **参数说明：**

  - bBoxes（aclTensor*，计算输入）：预测矩形框，shape为(m, 4)的二维tensor，m为bounding boxes的数量，4指[x0, y0, x1, y1]，(x0, y0)和(x1, y1)分别表示矩形框的左上角和右下角，需满足x1 > x0, y1 > y0。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>：FLOAT、FLOAT16
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、BFLOAT16
  - gtBoxes（aclTensor*，计算输入）：真值矩形框，shape为(n, 4)的二维tensor，n为bounding boxes的数量，4指[x2, y2, x3, y3]，(x2, y2)和(x3, y3)分别表示矩形框的左上角和右下角，需满足x3 > x2, y3 > y2。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据类型需要和bBoxes保持一致。
    - <term>Atlas 推理系列产品</term>：FLOAT、FLOAT16
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、BFLOAT16
  - mode（char*，计算输入）: 用于选择计算方式"iou"或"iof"。Host侧的字符串，数据类型支持String。
    - "iou"：计算交并比。
    - “iof"：计算前景交叉比。
  - eps（float，计算输入）：防止除零，计算面积时长和宽都会加上eps。Host侧的浮点型，数据类型支持FLOAT。
  - aligned（bool，计算输入）：用于标识两个输入的shape是否相同。Host侧的布尔型，数据类型支持BOOL。
    - True：bBoxes和gtBoxes的shape保持一致，都是(m, 4)，输出的shape为(m, 1)。
    - False：bBoxes和gtBoxes的shape不一致，分别是(m, 4)和(n, 4)，输出的shape为(m, n)。
  - overlap（aclTensor*，计算输出）：根据两个输入计算得到的交并比/前景交叉比，shape为(m, n)或(m, 1)的二维tensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据类型需要和bBoxes保持一致。
    - <term>Atlas 推理系列产品</term>：FLOAT、FLOAT16
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、BFLOAT16
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的bBoxes、gtBoxes和输出overlap是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. bBoxes、gtBoxes、overlap不是二维。
                                        2. bBoxes、gtBoxes、overlap的数据类型不一致。
                                        3. bBoxes、gtBoxes、overlap的数据类型和数据格式不在支持的范围内。
                                        4. bBoxes或gtBoxes的第二维不是4。
                                        5. aligned为true时，bBoxes和gtBoxes的第一维不相同。
                                        6. aligned为true时，overlap的第二维不是1。
                                        7. mode不是"iou"或"iof"。
                                        8. eps小于0。
  返回561103 (ACLNN_ERR_INNER_NULLPTR): 1. API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。
  返回361001 (ACLNN_ERR_RUNTIME_ERROR)：1. API调用npu runtime的接口异常，如SocVersion不支持。
  ```

## aclnnIou

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnIouGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnIou默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_iou.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> bBoxesHostData = {1.0, 1.0, 5.0, 3.0, 1.0, 1.0, 5.0, 3.0};
  std::vector<float> gtBoxesHostData = {4.0, 2.0, 9.0, 5.0, 4.0, 2.0, 9.0, 5.0};
  std::vector<float> overlapHostData = {0.045455, 0.045455};
  std::vector<int64_t> bBoxesShape = {2, 4};
  std::vector<int64_t> gtBoxesShape = {2, 4};
  std::vector<int64_t> overlapShape = {2, 1};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  const char* mode = "iou";
  float eps = 0.0f;
  bool aligned = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  ret = aclnnIouGetWorkspaceSize(bBoxes, gtBoxes, mode, eps, aligned, overlap, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnIouGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnIou
  ret = aclnnIou(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnIou failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(overlapShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), overlapDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);

  // 7. 释放device资源
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
