# ScaleAndTranslate

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

- 算子功能：按给定输出尺寸、缩放因子和平移量对输入图像执行二维重采样。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------|
| images   | 输入       | 输入为四维 Tensor，shape 为 (batch, input_height, input_width, channels)。 | INT8、UINT8、INT16、UINT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE | NHWC |
| size   | 输入       | 包含 2 个元素的一维 Tensor，依次表示输出高度和输出宽度 [output_height, output_width]，两个值都必须为正。 | INT32                                                   | ND   |
| scale   | 输入       | 包含 2 个元素的一维 Tensor，依次表示行、列方向的缩放因子 [row_scale, col_scale]，两个值都必须大于 0。 | FLOAT                                                   | ND   |
| translation   | 输入       | 包含 2 个元素的一维 Tensor，依次表示行、列方向的平移量 [row_translation, col_translation]。 | FLOAT                                                   | ND   |
| kernel_type   | 可选属性     | • 指定重采样核类型。支持 "lanczos1"，"lanczos3"，"lanczos5"，"gaussian"，"box"，"triangle"，"keyscubic"，"mitchellcubic"。<br>• 默认值为 "lanczos3"。 | STRING                                                  | -    |
| antialias   | 可选属性     | • 指定是否在缩小场景下按更大的有效核半径生成采样权重，以降低混叠。<br>• 默认值为 true。 | BOOL                                                    | -    |
| y   | 输出       | 返回四维 FLOAT Tensor，shape 为 (batch, output_height, output_width, channels)。 | FLOAT                                                   | NHWC |

## 约束说明

- images 必须为四维 Tensor，且 input_height、input_width、channels 必须大于 0。
- size 必须为包含 2 个元素的一维 INT32 Tensor，且 output_height、output_width 必须为正。
- scale 按源码实现读取前 2 个 FLOAT 元素，分别表示 row_scale 和 col_scale，且两者都必须大于 0。
- translation 按源码实现读取前 2 个 FLOAT 元素，分别表示 row_translation 和 col_translation。
- y 的数据类型固定为 FLOAT，shape 由 images 的 batch、channels 以及 size 指定的输出高宽共同决定。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_scale_and_translate](./examples/test_geir_scale_and_translate.cpp) | 通过[算子IR](./op_graph/scale_and_translate_proto.h)构图方式调用ScaleAndTranslate算子。 |