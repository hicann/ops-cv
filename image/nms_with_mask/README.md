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
| box_scores | 输入 | 二维 Tensor，其 shape 为 (num_boxes, 5)，其中 5 表示 [y1, x1, y2, x2, score]。 | FLOAT16、FLOAT、BF16 | ND |
| iou_threshold | 属性 | 浮点数，表示用于判断候选框是否在 IoU（交并比）上重叠过多的阈值。默认值为 0.5。 | FLOAT | - |
| selected_boxes | 输出 | 二维 Tensor，其 shape 为 (num_boxes, 5)，表示经过 NMS 筛选后的框及分数。 | FLOAT16、FLOAT、BF16 | ND |
| selected_idx | 输出 | 一维 Tensor，其 shape 为 (num_boxes)，表示被选中框的索引。 | INT32 | ND |
| selected_mask | 输出 | 一维 Tensor，其 shape 为 (num_boxes)，表示被选中框的有效性掩码。 | UINT8 | ND |


## 约束说明

- 输入的 box_scores 最后一维必须为 5，依次表示 [x1, y1, x2, y2, score]。
- 输出 selected_boxes、selected_idx、selected_mask 的 batch 维度与输入 box_scores 的 batch 维度保持一致。
- 在 FLOAT16 或 BF16 场景下，算子进行排序和计算对比标杆可能会引入计算误差。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/nms_with_mask_proto.h)构图方式调用NMSWithMask算子。         |
