# ImageProjectiveTransform

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入图像施加射影变换（Projective Transform），根据变换矩阵将输出图像中的每个像素映射回输入图像中对应的坐标，再通过插值计算输出像素值。

- 计算公式：

  对于输出图像中的每个像素 $(x_{out}, y_{out})$，其对应的输入图像坐标 $(x_{in}, y_{in})$ 通过射影变换矩阵计算：

  $$
  x_{in} = \frac{a_0 \cdot x_{out} + a_1 \cdot y_{out} + a_2}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  $$
  y_{in} = \frac{a_3 \cdot x_{out} + a_4 \cdot y_{out} + a_5}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  其中 $a_0, a_1, \ldots, a_7$ 为变换矩阵的8个参数，由输入 `transforms` 张量提供。

  根据插值模式计算输出像素值：

  - **BILINEAR（双线性插值）**：

    $$
    x_{floor} = \lfloor x_{in} \rfloor, \quad x_{ceil} = x_{floor} + 1
    $$

    $$
    y_{floor} = \lfloor y_{in} \rfloor, \quad y_{ceil} = y_{floor} + 1
    $$

    $$
    v_{yfloor} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{floor}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{floor})
    $$

    $$
    v_{yceil} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{ceil}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{ceil})
    $$

    $$
    result = (y_{ceil} - y_{in}) \cdot v_{yfloor} + (y_{in} - y_{floor}) \cdot v_{yceil}
    $$

    当源坐标超出输入图像边界时，使用填充值（默认为0）代替越界像素值。当变换矩阵分母为0或结果为NaN/Inf时，浮点类型输出NaN，整数类型输出INT_MIN。

  - **NEAREST（最近邻插值）**：

    $$
    x_i = \text{round}(x_{in}), \quad y_i = \text{round}(y_{in})
    $$

    当源坐标超出输入图像边界时，使用填充值（默认为0）。

## 参数说明

- **参数说明**：

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
      <td>images</td>
      <td>输入</td>
      <td>输入图像张量，shape为(N, H, W, C)，其中N为批次数，H为图像高度，W为图像宽度，C为通道数。</td>
      <td>FLOAT16 / FLOAT / UINT8 / INT32</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>transforms</td>
      <td>输入</td>
      <td>射影变换矩阵参数，shape为(N, 8)，每行包含8个变换参数[a0, a1, a2, a3, a4, a5, a6, a7]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_shape</td>
      <td>输入</td>
      <td>输出图像的空间尺寸，shape为(2,)，包含[height, width]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation</td>
      <td>属性（必选）</td>
      <td>插值模式，取值为"BILINEAR"（双线性插值）或"NEAREST"（最近邻插值）。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fill_mode</td>
      <td>属性（可选）</td>
      <td>填充模式，默认值为"CONSTANT"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transformed_images</td>
      <td>输出</td>
      <td>变换后的图像张量，shape为(N, HOut, WOut, C)，数据类型与images一致。</td>
      <td>FLOAT16 / FLOAT / UINT8 / INT32</td>
      <td>NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- images输入必须为4维（NHWC格式）。
- transforms输入必须为2维，shape为(N, 8)，数据类型为FLOAT。
- output_shape输入必须为1维，shape为(2,)，数据类型为INT32。
- fill_mode当前仅支持"CONSTANT"模式，填充值为0。

## 调用说明

无
