# Iou3D

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算两组3D旋转框（7-DoF：`[x, y, z, w, h, d, theta]`）之间的3D IoU矩阵。
- 计算原理：
  - BEV投影：每个框在XY平面按`theta`旋转，得到4个顶点：`P = center ± 0.5*w*(cos, sin) ± 0.5*h*(-sin, cos)`。
  - 交集面积：顶点包含测试 + 边相交，收集交集多边形顶点；0/3顶点直接算三角形面积，>3顶点质心分解 + 极角排序求和。
  - Z轴重叠：`real_d = max(min(z1max, z2max) - max(z1min, z2min), 0)`，其中`z_min = z - 0.5*d`，`z_max = z + 0.5*d`。
- 计算公式：

  $$
  \text{IoU}_{3D} = \frac{\text{bev\_area} \times \text{real\_d}}{V_A + V_B - V_{inter} + \varepsilon}, \quad \varepsilon = 10^{-6}
  $$

  其中`V = w * h * d`为框体积，`V_inter = bev_area * real_d`为交集体积。
- 数值稳定三守卫：分母加`epsilon = 1e-6`防除零；Z轴重叠clamp到非负；退化多边形（顶点 < 3）面积置0。

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
    <td>bboxes</td>
    <td>输入</td>
    <td>预测框，shape <code>[B, 7, N]</code>。第二维为7-DoF <code>[x, y, z, w, h, d, theta]</code>。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gtboxes</td>
    <td>输入</td>
    <td>真值框，shape <code>[B, 7, K]</code>。第二维为7-DoF <code>[x, y, z, w, h, d, theta]</code>。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>iou</td>
    <td>输出</td>
    <td>3D IoU矩阵，shape <code>[B, N, K]</code>，值域 <code>[0, 1]</code>。<code>iou[b, n, k]</code> = 第b批第n个预测框与第k个真值框的3D IoU。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

- 7-DoF语义：`x, y, z`为中心坐标，`w, h, d`为框在三个方向的尺寸，`theta`为绕Z轴的旋转角（弧度）。
- 三个Tensor的batch维`B`必须一致；`bboxes.shape[1] == gtboxes.shape[1] == 7`。

## 约束说明

- 数据类型：`bboxes`、`gtboxes`、`iou`均必须为float32。
- channel固定为7：`bboxes.shape[1]`与`gtboxes.shape[1]`必须等于7（7-DoF），否则拒绝。
- K 无上限：对标 mmcv，`gtboxes.shape[2]`（K）不设上限；逐对计算的UB与极角排序（固定32元素）缓冲与K无耦合，任意K成立。
- 无效框排序：同一batch内无效框（`w * h * d = 0`）不能排在有效框之前。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ------------ | ------------ | ------------ |
| GE图模式 | [test_geir_iou3d](./examples/test_geir_iou3d.cpp) | 通过[算子IR](./op_graph/iou3d_proto.h)构图方式调用Iou3D算子|
