# IMGWarpOffsets

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：该算子根据张量 offsets 对输入张量 images 图像进行偏移变换，将 images 转化为张量 warp_images。

## 参数说明

| 参数名                 | 输入/输出/属性 | 描述                                                                         | 数据类型                | 数据格式 |
|---------------------|----------|----------------------------------------------------------------------------|---------------------|------|
| images              | 输入       | 四维 Tensor，其 shape 为(batch, image_height, image_width, 3)。                  | UINT8、FLOAT16、FLOAT | ND   |
| offsets             | 输入       | 四维 Tensor，其 shape 为(batch, 4, new_height, new_width)。                      | FLOAT、INT32         | ND   |
| warp_images         | 输出       | 五维 Tensor，其 shape 为(batch, 4, new_height, new_width, 3)，数据类型与输入 images 相同。 | UINT8、FLOAT16、FLOAT             | ND   |


## 约束说明

- 当 offsets 的数据类型为 INT32 时，images 必须为 FLOAT16。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_image_warp_offsets](./examples/test_geir_image_warp_offsets.cpp)   | 通过[算子IR](./op_graph/image_warp_offsets_proto.h)构图方式调用IMGWarpOffsets算子。 |


