# aclnnRoiAlignV2

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_align)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

算子功能：RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。[aclnnRoiAlign](./aclnnRoiAlign.md)对标ONNX opset 10算子原型，aclnnRoiAlignV2对标torchvision算子原型。aclnnRoiAlignV2使用boxes替代aclnnRoiAlign的rois和batch_indices，并增加aligned入参，同时取消mode入参、默认执行mode="avg"场景。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRoiAlignV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRoiAlignV2”接口执行计算。

- `aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(const aclTensor* self, const aclTensor* boxes, int64_t pooledHeight, int64_t pooledWidth, float spatialScale, int64_t samplingRatio, bool aligned, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnRoiAlignV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRoiAlignV2GetWorkspaceSize

- **参数说明：**

  - self（aclTensor\*，计算输入）：图像特征图输入。Device侧的aclTensor，数据类型支持FLOAT16、FLOAT，必须与boxes、out数据类型一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持NCHW。维度为4维，shape为（B, C, inputHeight, inputWidth），表示输入张量一个batch内有B张图像，每个图像有C个尺寸为inputHeight \* inputWidth的特征图。B、inputHeight、inputWidth不支持0维。

  - boxes（aclTensor\*，计算输入）：感兴趣区域box坐标。Device侧的aclTensor，数据类型支持FLOAT16、FLOAT，必须与self、out数据类型一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。维度为2维，shape为（K, 5），5代表box相关信息（image_id, x1, y1, x2, y2），K需要与out第0维保持一致。image_id取值范围\[0, B\)，向下取整到图像id，B为self第0维大小。坐标满足0 <= x1 <= x2 <= inputWidth/spatialScale、0 <= y1 <= y2 <= inputHeight/spatialScale。

  - pooledHeight（int64_t，计算输入）：池化后输出图像的高度。Host侧的输入参数。

  - pooledWidth（int64_t，计算输入）：池化后输出图像的宽度。Host侧的输入参数。
  
  - spatialScale（float，计算输入）：乘法空间尺度因子，将ROI坐标从其输入空间尺度转换为池化时使用的尺度，即输入特征图X相对于输入图像的空间尺度。Host侧的输入参数，需大于0。

  - samplingRatio（int64_t，计算输入）：用于计算每个输出元素在H和W方向上的采样频率。Host侧的输入参数，需大于等于0。
  
  - aligned（bool，计算输入）：如果为false，则对齐[aclnnRoiAlign](./aclnnRoiAlign.md)版本实现；如果为true，则box坐标像素偏移-0.5来使相邻像素索引更好对齐。Host侧的输入参数。

  - out（aclTensor\*，计算输出）：池化后的输出。Device侧的aclTensor，数据类型支持FLOAT16、FLOAT，必须与self、boxes数据类型一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持NCHW。维度为4维，shape为（K，C，pooledHeight，pooledWidth），表示输出张量一个batch内有K个元素，每个元素有C个尺寸为pooledHeight \* pooledWidth的特征图。

  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。


- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001(ACLNN_ERR_PARAM_NULLPTR)：传入的self、boxes、out是空指针。
返回161002(ACLNN_ERR_PARAM_INVALID)：1. self、boxes和out的数据类型和数据格式不在支持的范围内。
                                    2. self、boxes、out的shape不满足约束限制。
                                    3. spatialScale需大于0，samplingRatio需大于等于0。
```

## aclnnRoiAlignV2

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnRoiAlignV2GetWorkspaceSize获取。

  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream，入参）：指定执行任务的Stream。


- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRoiAlignV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2.h"

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

template <typename T>
int CreateAclNchTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
  std::vector<int64_t> boxesShape = {1, 5};
  std::vector<int64_t> outShape = {1, 1, 3, 3};

  void* selfDeviceAddr = nullptr;
  void* boxesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* boxes = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<float> boxesHostData = {0.0, -2.0, -2.0, 22.0, 22.0};
  std::vector<float> outHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};

  // 创建self aclTensor
  ret = CreateAclNchTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建boxes aclTensor
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclNchTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int64_t pooledHeight = 3;
  int64_t pooledWidth = 3;
  int64_t samplingRatio = 2;
  float spatialScale = 0.25f;
  bool aligned = false;

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRoiAlignV2第一段接口
  ret = aclnnRoiAlignV2GetWorkspaceSize(self, boxes, pooledHeight, pooledWidth, spatialScale, 
                                              samplingRatio, aligned, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnRoiAlignV2第二段接口
  ret = aclnnRoiAlignV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(boxes);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(boxesDeviceAddr);
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
