# ResizeBicubicV2Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：计算输入图像在双三次插值基础下的梯度。
- 计算公式：
  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|2 + 1 & \text{for } |x|\leq1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1\lt|x|\lt2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  $$
  \frac{\alpha L}{\alpha X_{i,j}} = \sum_{i'} \sum_{j'} \frac{\alpha L}{\alpha Y_{i',i'}} \times W(i' - i) \times W(j' - j)
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
      <td>grads</td>
      <td>输入</td>
      <td>正向双三次插值调整后的图，对应公式Y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>输入</td>
      <td>原图像的高和宽。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向Resize的输入梯度。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- 无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_resize_bicubic_v2_grad](examples/test_aclnn_resize_bicubic_v2_grad.cpp) | 通过[aclnnUpsampleBicubic2dBackward](image/upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md)接口方式调用ResizeBicubicV2Grad算子。 |