# IMGWarpOffsets

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：该算子根据张量offsets对输入张量images进行偏移变换，生成warp_images张量。

## 参数说明

| 参数名                 | 输入/输出/属性 | 描述                                                                         | 数据类型                | 数据格式 |
|---------------------|----------|----------------------------------------------------------------------------|---------------------|------|
| images              | 输入       | 四维Tensor，其shape为(batch, image_height, image_width, 3)。                  | UINT8、FLOAT16、FLOAT | ND   |
| offsets             | 输入       | 四维Tensor，其shape为(batch, 4, new_height, new_width)。                      | FLOAT、INT32         | ND   |
| warp_images         | 输出       | 五维Tensor，其shape为(batch, 4, new_height, new_width, 3)，数据类型与输入images相同。 | UINT8、FLOAT16、FLOAT             | ND   |

## 约束说明

- 当offsets的数据类型为INT32时，images必须为FLOAT16。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_image_warp_offsets](./examples/test_geir_image_warp_offsets.cpp)   | 通过[算子IR](./op_graph/image_warp_offsets_proto.h)构图方式调用IMGWarpOffsets算子。 |
