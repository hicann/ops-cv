# ExtractGlimpseV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：从批量输入图像中提取指定位置和大小的子图像（glimpse）。根据给定的偏移坐标和裁剪尺寸，从每张输入图像中裁剪出一个子区域，输出为一批裁剪后的图像。

- 计算公式：

$$
\text{对于每个 batch item } i:
$$
$$
\text{1. 坐标变换: } offset\_y, offset\_x \text{ (根据 normalized/centered 属性)}
$$
$$
\text{2. 裁剪区域: } [start\_y, end\_y) \times [start\_x, end\_x) \text{ (clamp 到图像边界)}
$$
$$
\text{3. 数据拷贝: } glimpse[i][base\_y:base\_y+copy\_h][base\_x:base\_x+copy\_w][:] = input[i][start\_y:end\_y][start\_x:end\_x][:]
$$
$$
\text{4. 越界填充: } glimpse \text{ 中未拷贝区域填零 (noise="zero")}
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
      <td>input</td>
      <td>输入</td>
      <td>输入图像批次，4D tensor，shape为[batch, height, width, channels]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>裁剪尺寸[glimpse_h, glimpse_w]，1D const tensor，shape为[2]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>每个batch item的偏移坐标(y, x)，2D tensor，shape为[batch, 2]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>glimpse</td>
      <td>输出</td>
      <td>裁剪后的图像批次，4D tensor，shape为[batch, size_h, size_w, channels]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>centered</td>
      <td>属性</td>
      <td>偏移坐标是否相对于图像居中。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normalized</td>
      <td>属性</td>
      <td>偏移坐标是否归一化到[-1, 1]。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>uniform_noise</td>
      <td>属性</td>
      <td>是否使用均匀分布噪声（当前实现仅支持false）。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>属性</td>
      <td>噪声类型（当前实现仅支持"zero"）。默认值："uniform"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- input 必须为 4D tensor。
- offsets 必须为 2D tensor，且第二维必须为 2。
- size 必须为 1D const tensor，且长度为 2。
- input 和 offsets 的 batch 维度必须一致。
- 当前仅支持 float32 数据类型。
- 当前仅支持 noise="zero" 的越界填充模式。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/extract_glimpse_v2_proto.h)构图方式调用ExtractGlimpseV2算子。         |
