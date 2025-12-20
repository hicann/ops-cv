# CropAndResize

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：从输入图像中提取多个裁剪区域，并将它们统一调整为指定大小，支持双线性插值和最近邻插值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                                                                                                                                                                                                                                                         | 数据类型                                                    | 数据格式 |
|-----|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------|
| x   | 输入       | 输入为四维的Tensor，其shape为(batch, image_height, image_width, depth)，其中image_height和image_width必须是正数。                                                                                                                                                                                             | UINT8、UINT16、INT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE | NHWC |
| boxes   | 输入       | 二维Tensor，其shape为(num_boxes, 4)。第i行指定了box_index[i]图像中一个框的位置，并且使用归一化坐标[y1, x1, y2, x2]进行表示。归一化后的y坐标值通过y*(image_height - 1)映射到图像坐标系中，因此归一化图像的高度[0, 1]区间会被映射到原始图像高度的[0，image_height - 1]区间。我们允许y1 > y2, 在这种情况下，采样的裁剪区域会是原始图像的上下翻转版。宽度维度的处理方式相似，归一化坐标值超出[0, 1]范围时，使用extrapolation_value外插值进行补齐。 | FLOAT16、FLOAT                                           | ND   |
| box_index   | 输入       | boxes的索引，其shape为(num_boxes)的一维tensor，取值范围[0, batch)，box_index[i]的值指定了第i个框所对应的图像。                                                                                                                                                                                                           | INT32                                                   | ND   |
| crop_size   | 输入       | 包含2个元素的1-D的Tensor，shape为(crop_height, crop_width)，所有裁剪后的图像区域都会被调整为此大小。图像内容的长宽比不会被保留。crop_height和crop_width必须是正数。                                                                                                                                                                           | INT32                                                   | ND   |
| extrapolation_value   | 可选属性     | • 可选外插值。<br>• 默认值为 0.0。                                                                                                                                                                                                                                                                    | FLOAT                                                   | -    |
| method   | 可选属性     | • 指定调整大小时的采样方法，为可选字符串。提供的方法有：“bilinear”，“nearest”，“bilinear_V2”。<br>• 默认值为“bilinear”。                                                                                                                                                                                                        | STRING                                                  | -    |
| y   | 输出       | 返回一个四维Tensor，shape大小为(num_boxes, crop_height, crop_width, depth)。                                                                                                                                                                                                                          | FLOAT16、FLOAT                                           | NHWC |


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_crop_and_resize](./examples/test_geir_crop_and_resize.cpp)   | 通过[算子IR](./op_graph/crop_and_resize_proto.h)构图方式调用CropAndResize算子。 |


