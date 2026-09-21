# Yuv444ToYuv422

## 贡献说明

| 贡献者      | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------|------------|----------|
| CANN-BOT SIMT | yuv444_to_yuv422 | 2026/06/17 | 从ops-math迁移yuv444_to_yuv422算子到 ops-cv |

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    ×     |
| <term>Atlas A2系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：将YUV444格式图像数据转换为YUV422格式。YUV444输入每像素包含4通道(Y, U, Y', V)，对水平相邻像素对的色度分量(U, V)进行2:1子采样（取平均值），输出YUV422格式每像素2通道(Y, UV)。

- 计算公式：

$$
y[i, j, 0] = \text{clip\_uint8}(Y_0)
$$
$$
y[i, j, 1] = \text{clip\_uint8}((U_0 + U_1) / 2)
$$
$$
y[i, j+1, 0] = \text{clip\_uint8}(Y'_0)
$$
$$
y[i, j+1, 1] = \text{clip\_uint8}((V_0 + V_1) / 2)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>输入</td>
      <td>YUV444输入图像数据，shape为(h, w, 4)，每像素4通道 (Y, U, Y', V)。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>YUV422输出图像数据，shape为(h, w, 2)，每像素2通道(Y, UV)。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入必须为3维张量，第三维固定为4（YUV444的4通道打包格式）。
- 输出为3维张量，前两个维度与输入相同，第三维固定为2（YUV422的2通道打包格式）。
- 仅支持float16输入、uint8输出。
- 仅支持Ascend 950PR&950DT系列产品。
- 输入输出要求连续存储（ND layout）。
