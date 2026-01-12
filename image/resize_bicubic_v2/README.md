# ResizeBicubicV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |

## 功能说明

- 算子功能：使用双三次插值调整图像大小到指定的大小。
- 计算公式：  
  周边16个点的像素位置：
  
  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|2 + 1 & \text{for } |x|\leq1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1\lt|x|\lt2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  像素值：

  $$
  B(X,Y) = \sum_{i=0}^3 \sum_{j=0}^3a_{ij} \times W(i) \times W(j)
  $$


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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中x。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>双三次插值调整后图像。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_resize_bicubic_v2](examples/test_aclnn_bicubic_v2.cpp) | 通过[aclnnUpsampleBicubic2d](../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)接口方式调用ResizeBicubicV2算子。 |