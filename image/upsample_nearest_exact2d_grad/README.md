# UpsampleNearestExact2dGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[UpsampleNearest](../upsample_nearest/README.md)在exact_mode为true时的反向传播。
- 计算公式：

  对于输入gradOut(N, C, h, w)，输出gradInput上任意一点(N, C, H, W)，则有：

  $$
  gradInput(N, C, H, W) = \sum_{h = srcH}^{srcHUp - 1}\sum_{w = srcW}^{srcWUp - 1}gradOut(N, C, h, w)
  $$

  其中：

  $$
  scalesH = outputSize[0]/inputSize[2]
  $$

  $$
  scalesW = outputSize[1]/inputSize[3]
  $$

  $$
  srcH = Min(ceil(scalesH * H - 0.5), outputSize[0])
  $$

  $$
  srcHUp = Min(ceil(scalesH * (H + 1) - 0.5), outputSize[0])
  $$

  $$
  srcW = Min(ceil(scalesW * W - 0.5), outputSize[1])
  $$

  $$
  srcWUp = Min(ceil(scalesW * (W + 1) - 0.5), outputSize[1])
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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式中的`grad_output`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>表示输入`grad_output`在H和W维度上的空间大小。size为2，且各元素均大于零。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td>
      <td>表示输出`grad_input`分别在N、C、H和W维度上的空间大小。size为4，且各元素均大于零。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>表示输出`grad_input`的height维度乘数，对应公式中的`scales_h`。不能传入负值。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>表示输出`grad_input`的width维度乘数，对应公式中的`scales_w`。不能传入负值。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`grad_input`。数据类型和数据格式与入参`grad_output`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest_exact1d_backward](examples/test_aclnn_upsample_nearest_exact1d_backward.cpp) | 通过[aclnnUpsampleNearestExact1dBackward](docs/aclnnUpsampleNearestExact1dBackward.md)接口方式调用UpsampleNearestExact2dGrad算子。 |
| aclnn接口  | [test_aclnn_upsample_nearest_exact2d_grad](examples/test_aclnn_upsample_nearest_exact2d_grad.cpp) | 通过[aclnnUpsampleNearestExact2dBackward](docs/aclnnUpsampleNearestExact2dBackward.md)接口方式调用UpsampleNearestExact2dGrad算子。 |
