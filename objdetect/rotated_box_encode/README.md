# RotatedBoxEncode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

将参考旋转框（anchor_box）与对应的 ground-truth 旋转框（gt_box）之间的几何偏差编码为 5 通道回归 delta 目标 `(dx, dy, dw, dh, dθ)`，用于旋转目标检测训练阶段的回归监督。输入与输出均为 3D 张量 `(B, 5, N)`，5 通道按角点形式 corner form `(x0, y0, x1, y1, θ_deg)` 组织，`θ_deg` 为角度制。设 `weight = (wx, wy, ww, wh, wa)`，逐通道编码公式为：

$$
\begin{aligned}
w_a &= \max(x1_a - x0_a,\ 1.0),\quad h_a = \max(y1_a - y0_a,\ 1.0) \\
cx_a &= x0_a + w_a / 2,\quad cy_a = y0_a + h_a / 2 \\
w_g &= \max(x1_g - x0_g,\ 1.0),\quad h_g = \max(y1_g - y0_g,\ 1.0) \\
cx_g &= x0_g + w_g / 2,\quad cy_g = y0_g + h_g / 2 \\
dx &= \frac{cx_g - cx_a}{w_a} \cdot wx \\
dy &= \frac{cy_g - cy_a}{h_a} \cdot wy \\
dw &= \ln\frac{w_g}{w_a} \cdot ww \\
dh &= \ln\frac{h_g}{h_a} \cdot wh \\
d\theta &= \bigl(\tan(\theta_g \cdot \pi/180) - \tan(\theta_a \cdot \pi/180)\bigr) \cdot wa
\end{aligned}
$$

其中 `max(·, 1.0)` 防止退化框导致除零与 `log(0)`，并按 `np.maximum` 语义传播 NaN。fp16 输入在内部升精度到 fp32 完成对数、三角与除法计算后再回写 fp16。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>anchor_box</td>
<td>输入</td>
<td>表示anchor参考旋转框，对应公式中anchor_box。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>gt_box</td>
<td>输入</td>
<td>表示ground-truth旋转框，对应公式中gt_box。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>表示编码后的5通道delta，对应公式中(dx,dy,dw,dh,dθ)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>weight</td>
<td>可选属性</td>
<td>5通道编码权重[wx,wy,ww,wh,wa]，默认[1.0,1.0,1.0,1.0,1.0]。</td>
<td>ListFloat</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- anchor_box、gt_box、y 三者数据类型必须一致，且仅支持 FLOAT16 或 FLOAT。
- anchor_box 与 gt_box 的 shape 必须完全一致（无广播），秩为 3，且第 1 维必须为 5。
- weight 若提供，长度必须为 5；不提供时使用默认值 `[1.0, 1.0, 1.0, 1.0, 1.0]`。
- 空Tensor（B=0 或 N=0）直接短路返回空输出，不报错。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式 | [test_geir_rotated_box_encode](./examples/arch35/test_geir_rotated_box_encode.cpp) | 通过[算子IR](./op_graph/rotated_box_encode_proto.h)构图方式调用RotatedBoxEncode算子。 |
