# BoundingBoxEncode

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×   |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算锚框（anchor box）与真实边界框（ground truth box）之间的编码偏移量，生成目标检测回归目标。

- 计算公式：

  先将输入坐标 $(x_1, y_1, x_2, y_2)$ 转换为中心点+宽高格式：

  $$
  cx = (x_1 + x_2) / 2, \quad cy = (y_1 + y_2) / 2, \quad w = x_2 - x_1 + 1, \quad h = y_2 - y_1 + 1
  $$

  再计算编码偏移量：

  $$
  dx = \frac{g_{cx} - p_{cx}}{p_w}, \quad dy = \frac{g_{cy} - p_{cy}}{p_h}, \quad dw = \ln\left(\frac{g_w}{p_w}\right), \quad dh = \ln\left(\frac{g_h}{p_h}\right)
  $$

  最后做均值标准化：

  $$
  \delta_i = \frac{raw_i - means_i}{stds_i}, \quad i \in \{0,1,2,3\}
  $$

  其中$p$为anchor_box对应值，$g$为ground_truth_box对应值。

## 参数说明

<table style="table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 350px">
  <col style="width: 200px">
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
      <td>anchor_box</td>
      <td>输入</td>
      <td>锚框坐标张量，坐标格式为(x1, y1, x2, y2)。数据类型需与ground_truth_box一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ground_truth_box</td>
      <td>输入</td>
      <td>真实边界框坐标张量，坐标格式为(x1, y1, x2, y2)。数据类型和shape需与anchor_box一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>means</td>
      <td>属性</td>
      <td>编码均值偏移量，长度为4。默认值为[0.0, 0.0, 0.0, 0.0]。</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stds</td>
      <td>属性</td>
      <td>编码标准差缩放量，长度为4，各元素不可为0。默认值为[1.0, 1.0, 1.0, 1.0]。</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>delats</td>
      <td>输出</td>
      <td>编码偏移量输出张量。数据类型与anchor_box一致，shape与anchor_box相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- anchor_box和ground_truth_box的数据类型必须相同，支持float16和float32。
- anchor_box和ground_truth_box的shape必须完全一致，均为(N, 4)。
- means和stds的长度必须为4，stds各元素不可为0。
- 坐标格式为标准(x1, y1, x2, y2)格式，即左上角和右下角坐标。
- 公式中宽高计算包含+1偏移（w = x2 - x1 + 1, h = y2 - y1 + 1），保证宽高至少为1，防止除零。
- 支持空Tensor（N=0时返回空输出）。

## 调用说明

| 调用方式   | 样例代码                                                                        | 说明                                         |
| ---------------- |-----------------------------------------------------------------------------| --------------------------------------------------- |
| 图模式 | [test_geir_bounding_box_encode](./examples/arch35/test_geir_bounding_box_encode.cpp) | 通过[算子IR](./op_graph/bounding_box_encode_proto.h)构图方式调用BoundingBoxEncode算子。         |
