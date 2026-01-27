# ResizeLinearGrad

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

## 功能说明

- 算子功能：计算输入图像在双三次插值基础下的梯度。
- 计算公式：

  $$
  \frac{\alpha L}{\alpha X_i} = \frac{\alpha L}{\alpha Y_i} \times \frac{y_1 - y_0}{x_1 - x_0}
  $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 300px">
    <col style="width: 250px">
    <col style="width: 150px">
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
      <td>grads</td>
      <td>输入</td>
      <td>正向单线性插值调整后的图，对应公式Y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>输入</td>
      <td>原图像的高和宽。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向Resize的输入梯度。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 无

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| aclnn接口  | [test_aclnn_resize_linear_grad](examples/test_aclnn_resize_linear_grad.cpp) | 通过[aclnnUpsampleLinear1dBackward](image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md)接口方式调用ResizeLinearGrad算子。 |