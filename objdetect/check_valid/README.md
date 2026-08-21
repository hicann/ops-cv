# CheckValid

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

检查候选框是否在图像有效范围内。给定候选框`bbox_tensor=(x0, y0, x1, y1)`与图像元信息`img_metas=(H, W, r)`，判断每个框是否满足：

$$
\text{valid} = (x_0 \geq 0) \wedge (y_0 \geq 0) \wedge (x_1 \leq W \cdot r - 1) \wedge (y_1 \leq H \cdot r - 1)
$$

输出`valid_tensor`为int8类型，1表示合法、0表示非法。

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
<td>bbox_tensor</td>
<td>输入</td>
<td>候选框坐标(x0,y0,x1,y1)，shape=(N,4)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>img_metas</td>
<td>输入</td>
<td>图像元信息(H,W,r)，至少3个元素，前3个有效。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>valid_tensor</td>
<td>输出</td>
<td>合法性掩码，1=合法/0=非法，shape=(N,1)。</td>
<td>INT8</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- bbox_tensor的shape必须为(N, 4)，末维固定为4。
- bbox_tensor与img_metas的dtype必须一致，支持float16/float32。
- img_metas的元素数必须≥3（前3个为H, W, r）。
- 输入仅支持ND数据格式。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 180px">
<col style="width: 200px">
<col style="width: 620px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE图模式</td>
<td><a href="examples/arch35/test_geir_check_valid.cpp">test_geir_check_valid.cpp</a></td>
<td>通过<a href="op_graph/check_valid_proto.h">算子IR</a>构图方式调用CheckValid算子</td>
</tr>
</tbody>
</table>
