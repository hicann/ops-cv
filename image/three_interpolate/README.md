# ThreeInterpolate

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

- 算子功能：基于3个最近邻的加权线性特征插值。给定已知特征点的特征描述符、待插值点的3个最近邻索引和对应权重，对每个待插值点的每个特征通道，计算3个最近邻特征的加权和。

- 计算公式：

$$
y[b, n, c] = \sum_{k=0}^{2} weight[b, n, k] \times features[b, idx[b, n, k], c]
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
      <td>features</td>
      <td>输入</td>
      <td>已知特征点集合，shape=(B, M, C)。B=batch, M=已知点数, C=通道数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输入</td>
      <td>3个最近邻索引，shape=(B, N, 3)。N=待插值点数，每行3个索引值，取值范围[0, M)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>3个最近邻权重，shape=(B, N, 3)。与idx对应，通常由距离倒数归一化得到。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>插值后的特征，shape=(B, N, C)。B=batch, N=待插值点数, C=通道数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- features 和 weight 必须同 dtype（均为 float32 或均为 float16）。
- idx 可独立选择 int32 或 int64。
- 所有输入 tensor 必须为连续（contiguous）格式。
- idx 值必须在 [0, M) 范围内，由调用方保证不越界。
- 维度上限：B、M、C、N 均不超过 2^32-1，且 N×C 不超过 2^32-1，B×N×3 与 B×M×C 不超过 2^64-1。
- 支持空 tensor（shape 含 0 维），此时输出为空 tensor，不执行计算。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式</td>
    <td><a href="./examples/arch35/test_geir_three_interpolate.cpp">test_geir_three_interpolate</a></td>
    <td>通过<a href="./op_graph/three_interpolate_proto.h">算子IR</a>构图方式调用ThreeInterpolate算子，参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成编译和验证。</td>
  </tr>
</tbody>
</table>
