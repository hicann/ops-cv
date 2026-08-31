# RotatedOverlaps

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin X90 处理器系列产品</term> | × |
| <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：计算两组二维旋转框之间的交叠面积矩阵，输出交叠面积，不计算IoU。
- 计算原理：将每个旋转框转换为四个顶点，收集一组框的顶点位于另一组框内的点以及边相交点，按顶点顺序计算交集多边形面积。
- 坐标格式：当`trans=false`时，矩形框格式为`[x, y, w, h, theta]`；当`trans=true`时，矩形框格式为`[x1, y1, x2, y2, theta]`。
- `theta`的单位为度，输出为交叠面积而不是IoU。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
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
    <td>第一组二维旋转框，shape为<code>[B, 5, N]</code>。第二维为5，表示[x, y, w, h, theta]或[x1, y1, x2, y2, theta]。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query_boxes</td>
    <td>输入</td>
    <td>第二组二维旋转框，shape为<code>[B, 5, K]</code>。第二维为5，表示[x, y, w, h, theta]或[x1, y1, x2, y2, theta]。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>overlaps</td>
    <td>输出</td>
    <td>两组旋转框的交叠面积矩阵，shape为<code>[B, N, K]</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>trans</td>
    <td>属性</td>
    <td>是否使用[x1, y1, x2, y2, theta]坐标格式，默认值为false。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 两个输入的batch维`B`必须一致。
- 首版实现要求`B`、`N`和`K`为正数，且`query_boxes.shape[2]`（K）不超过2000。
- 输入框中存在非有限坐标或退化矩形时，该矩形框参与的配对输出为0。
- 当前不支持GE IR动态Rank场景（输入Shape声明为`[-2]`）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ------------ | ------------ | ------------ |
| GE图模式 | [test_geir_rotated_overlaps](./examples/test_geir_rotated_overlaps.cpp) | 通过[算子IR](./op_graph/rotated_overlaps_proto.h)构图方式调用RotatedOverlaps算子。 |
