# SortedNMS

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×   |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：在已按分数降序排列的候选框序列上，按照交并比（IoU）阈值贪心选择非抑制框，输出被选中框在原始boxes中的索引。

- 计算公式：

  $$
  IoU = \frac {Area_{inter}} {Area_{current} + Area_{next} - Area_{inter}}
  $$

  其中，Area_current为当前选中框的面积，Area_next为候选框的面积，Area_inter为两个框的重叠面积，offset为坐标计算偏移量。

  $$
  Area_i = max(X_{2i} - X_{1i} + offset, 0) * max(Y_{2i} - Y_{1i} + offset, 0) \\
  Area_{inter} = max(min(X_{2c}, X_{2n}) - max(X_{1c}, X_{1n}) + offset, 0) * max(min(Y_{2c}, Y_{2n}) - max(Y_{1c}, Y_{1n}) + offset, 0)
  $$

  算子按照sorted_scores的非递增顺序遍历候选框。当候选框的分数大于score_threshold且未被抑制时，将input_indices中对应的索引加入输出；当候选框与当前选中框的IoU大于iou_threshold时，抑制该候选框。

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
      <td>候选矩形框，shape为(N, 4)，坐标格式为(X1, Y1, X2, Y2)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_scores</td>
      <td>输入</td>
      <td>候选矩形框的分数，shape为(N,)，需要按非递增顺序排列。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_indices</td>
      <td>输入</td>
      <td>sorted_scores对应的候选框索引，shape为(N)，取值范围为[0, N)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_output_size</td>
      <td>输入</td>
      <td>最多输出的候选框数量，输入为标量或shape为(1,)的张量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>输入</td>
      <td>判断候选框是否需要抑制的IoU阈值，输入为标量或shape为(1,)的张量。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>输入</td>
      <td>过滤候选框的分数阈值，输入为标量或shape为(1,)的张量。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>属性</td>
      <td>计算坐标差值时使用的偏移量，取值为0或1，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selected_indices</td>
      <td>输出</td>
      <td>被选中候选框在原始boxes中的索引，shape为(M)，M为运行时计算结果且M不大于min(max_output_size, N)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- boxes与iou_threshold的数据类型需要保持一致；sorted_scores与score_threshold的数据类型需要保持一致，两组数据类型可以不同。

## 约束说明

- 输入shape限制：boxes为(N, 4)的二维张量，sorted_scores和input_indices为(N,)的一维张量，max_output_size、iou_threshold和score_threshold为标量或shape为(1,)的张量。

- sorted_scores需要按照非递增顺序排列，input_indices需要为合法的候选框索引。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_sorted_nms](./examples/test_geir_sorted_nms.cpp) | 通过[算子IR](./op_graph/sorted_nms_proto.h)构图方式调用SortedNMS算子。 |
