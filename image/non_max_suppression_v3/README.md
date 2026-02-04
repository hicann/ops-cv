# NonMaxSuppressionV3

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

- 算子功能：按照分数递减顺序，采用贪心策略选择候选框（bounding boxes）子集。

## 参数说明

| 参数名               | 输入/输出/属性 | 描述                                                                                | 数据类型          | 数据格式 |
|-------------------|----------|-----------------------------------------------------------------------------------|---------------|------|
| boxes             | 输入       | 二维 Tensor，其 shape 为 (num_boxes, 4)。输入格式为 (x1, y1, x2, y2)，并且要求 x1 < x2 且 y1 < y2。 | FLOAT16、FLOAT | ND   |
| scores            | 输入       | 一维 Tensor，其 shape 为 (num_boxes)。表示与每个候选框对应的分数（与 boxes 的每一行一一对应）。                  | FLOAT16、FLOAT | ND   |
| max_output_size   | 输入       | 标量整数 Tensor，表示非极大值抑制（Non-Maximum Suppression）最多选择的候选框数量。                          | INT32         | ND   |
| iou_threshold     | 输入       | 标量浮点 Tensor，表示用于判断候选框是否在 IoU（交并比）上重叠过多的阈值。                                        | FLOAT16、FLOAT | ND   |
| score_threshold   | 输入       | 标量浮点 Tensor，表示用于根据分数移除候选框的阈值。                                                     | FLOAT16、FLOAT         | ND   |
| offset            | 可选属性     | • 可选整数。<br>• 默认值为 0。                                                              | INT           | -    |
| selected_indices  | 输出       | 一维整数 Tensor，shape 为 (M)，表示从输入 boxes Tensor 中选中的索引，其中 M <= max_output_size。        | INT32 | ND   |


## 约束说明

- 输入的 boxes 和 scores 必须是 float 类型。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_non_max_suppression_v3](./examples/test_geir_non_max_suppression_v3.cpp)   | 通过[算子IR](./op_graph/non_max_suppression_v3_proto.h)构图方式调用NonMaxSuppressionV3算子。 |


