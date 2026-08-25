# NonMaxSuppressionV7

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|
|<term>Kirin X90 处理器系列产品</term>|×|
|<term>Kirin 9030 处理器系列产品</term>|×|

## 功能说明

- 算子功能：按batch和类别对候选框执行贪心非极大值抑制，输出被选中框的索引三元组(batch_index, class_index, box_index)。

- 计算公式：

  $$
  IoU = \frac {Area_{inter}} {Area_{current} + Area_{next} - Area_{inter}}
  $$

  候选框按分数降序选择。仅处理分数大于score_threshold的候选框；当候选框与已选框的IoU大于iou_threshold时，抑制该候选框。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>boxes</td>
      <td>输入</td>
      <td>候选框，shape为(B, N, 4)。center_point_box为0时坐标格式为(y1, x1, y2, x2)；center_point_box为1时坐标格式为(x_center, y_center, width, height)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scores</td>
      <td>输入</td>
      <td>候选框分数，shape为(B, C, N)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_output_size</td>
      <td>可选输入</td>
      <td>每个batch和类别最多输出的候选框数量，输入为标量或shape为(1,)的张量；小于等于0时不输出该类别。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>可选输入</td>
      <td>判断候选框是否需要抑制的IoU阈值，输入为标量或shape为(1,)的张量，默认值为0.0。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>可选输入</td>
      <td>过滤候选框的分数阈值，输入为标量或shape为(1,)的张量，默认值为0.0。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_id</td>
      <td>可选输入</td>
      <td>候选框的索引映射，shape为(B, C, N, 3)或(B, C, N, 4)。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>center_point_box</td>
      <td>属性</td>
      <td>候选框坐标格式，取值为0或1，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_boxes_size</td>
      <td>属性</td>
      <td>输出selected_indices第一维的大小，取值范围为[0, INT32_MAX]，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selected_indices</td>
      <td>输出</td>
      <td>被选中候选框的索引，shape为(max_boxes_size, 3)，不足部分填充(-1, -1, -1)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- boxes与scores的数据类型可以独立选择。

## 约束说明

- boxes和scores必须为静态正shape，且boxes.shape[0]等于scores.shape[0]、boxes.shape[1]等于scores.shape[2]。

- center_point_box仅支持0和1，max_boxes_size取值范围为[0, INT32_MAX]。

- max_output_size、iou_threshold和score_threshold为标量或shape为(1,)的张量；index_id为(B, C, N, 3)或(B, C, N, 4)的张量。

## 调用说明

|调用方式|调用样例|说明|
|:---|:---|:---|
|图模式调用|[test_geir_non_max_suppression_v7](./examples/arch35/test_geir_non_max_suppression_v7.cpp)|通过[算子IR](./op_graph/non_max_suppression_v7_proto.h)构图方式调用NonMaxSuppressionV7算子。|
