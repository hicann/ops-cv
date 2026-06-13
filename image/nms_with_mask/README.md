# NMSWithMask

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对边界框进行非极大值抑制（NMS）处理，输出经过NMS后的选中的框、索引以及掩码。用于目标检测后处理，去除重复检测的框。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------------------------|:----------:|:-----|:-----|:----:|
| box_scores | 输入 | 二维Tensor，其shape为(num_boxes, 5)，其中5表示[y1, x1, y2, x2, score]，num_boxes不超过39936。 | FLOAT16、FLOAT、BF16 | ND |
| iou_threshold | 属性 | 浮点数，表示用于判断候选框是否在IoU（交并比）上重叠过多的阈值。默认值为0.5。 | FLOAT | - |
| selected_boxes | 输出 | 二维Tensor，其shape为(num_boxes, 5)，和原输入的box_scores一样。 | FLOAT16、FLOAT、BF16 | ND |
| selected_idx | 输出 | 一维Tensor，其shape为(num_boxes)，表示0到num_boxes - 1的序列数。 | INT32 | ND |
| selected_mask | 输出 | 一维Tensor，其shape为(num_boxes)，表示目标框的掩码情况。 | UINT8 | ND |

## 约束说明

- 输入的box_scores最后一维必须为5，依次表示 [y1, x1, y2, x2, score]。
- 输出selected_boxes、selected_idx、selected_mask的batch维度与输入box_scores的batch维度保持一致。
- 在FLOAT16或BF16场景下，算子进行排序和计算对比标杆可能会引入计算误差。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/nms_with_mask_proto.h)构图方式调用NMSWithMask算子。         |
