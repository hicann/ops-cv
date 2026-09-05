# CombinedNonMaxSuppression

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对每个batch、每个类别独立执行贪心非极大值抑制（NMS），再按置信度从高到低合并各类别的候选框。
- 输入输出尺寸：

  - `boxes`：shape为(batch, num_boxes, q, 4)，其中q为1或num_classes。
  - `scores`：shape为(batch, num_boxes, num_classes)。
  - `nmsed_boxes`：shape为(batch, output_size, 4)。
  - `nmsed_scores`、`nmsed_classes`：shape为(batch, output_size)。
  - `valid_detections`：shape为(batch)。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--|:--|:--|:--|:--|
| boxes | 输入 | 候选框坐标，shape为(batch, num_boxes, q, 4)，最后一维依次表示y1、x1、y2、x2。 | FLOAT32 | ND |
| scores | 输入 | 候选框分数，shape为(batch, num_boxes, num_classes)。 | FLOAT32 | ND |
| max_output_size_per_class | 输入 | 每个类别最多保留的候选框数量，取值范围为[1, 1000]。 | INT32 | ND |
| max_total_size | 输入 | 每个batch最多输出的候选框数量，取值范围为[1, 1000]。 | INT32 | ND |
| iou_threshold | 输入 | IoU抑制阈值，取值范围为[0, 1]。 | FLOAT32 | ND |
| score_threshold | 输入 | 分数阈值，仅保留分数严格大于该值的候选框。 | FLOAT32 | ND |
| pad_per_class | 可选属性 | 是否按类别填充输出。为true时，output_size为min(max_total_size, max_output_size_per_class * num_classes)；默认为false。 | BOOL | - |
| clip_boxes | 可选属性 | 是否将输出候选框坐标裁剪到[0, 1]；默认为true。 | BOOL | - |
| nmsed_boxes | 输出 | 筛选后的候选框。 | FLOAT32 | ND |
| nmsed_scores | 输出 | 筛选后的候选框分数。 | FLOAT32 | ND |
| nmsed_classes | 输出 | 筛选后的类别索引，以浮点数表示。 | FLOAT32 | ND |
| valid_detections | 输出 | 每个batch中有效候选框的数量。 | INT32 | ND |

## 约束说明

- `boxes`必须为4维张量，`scores`必须为3维张量，且二者的batch和num_boxes维必须一致。
- `boxes`最后一维必须为4，q维必须为1或num_classes。
- batch、num_boxes和num_classes必须大于0；num_boxes不能超过200000，num_classes不能超过200。
- batch与num_classes的乘积不能超过`INT32_MAX`。
- `max_output_size_per_class`和`max_total_size`必须为标量，取值范围均为[1, 1000]。
- `iou_threshold`和`score_threshold`必须为标量，`iou_threshold`取值范围为[0, 1]。
- `max_output_size_per_class`、`max_total_size`、`iou_threshold`和`score_threshold`为值依赖输入，输入值必须在编译期已知；其中前两者的值参与输出Shape推导，后两者的值参与NMS计算。
- 候选框坐标顺序为(y1, x1, y2, x2)，支持反向坐标；计算IoU时会分别取坐标端点的最小值和最大值。
- 当前不支持GE IR动态Shape和动态Rank场景（输入Shape声明分别为`[-1]`和`[-2]`）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:--|:--|:--|
| 图模式 | [test_geir_combined_non_max_suppression](examples/arch35/test_geir_combined_non_max_suppression.cpp) | 通过[算子IR](op_graph/combined_non_max_suppression_proto.h)构图方式调用CombinedNonMaxSuppression算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。 |
