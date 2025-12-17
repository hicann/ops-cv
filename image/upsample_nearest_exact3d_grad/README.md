# UpsampleNearestExact3dGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[UpsampleNearestExact3d](../upsample_nearest_exact3d/README.md)的反向计算。
- 计算公式：
  
  $$
  gradInput(N, C, D, H, W) += gradOutput( N, C, ceil ( scales\_d * D - 0.5 ), ceil ( scales\_h * H - 0.5 ),  ceil ( scales\_w * W - 0.5 ))
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
      <td>表示反向计算的梯度Tensor，对应公式中的输入`gradOutput`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td>
      <td>表示输出`y`分别在N、C、D、H和W维度上的空间大小。size为5，且各元素均大于零。必须满足：input_size[0] == grad_output_tensor_size[0]；input_size[1] == grad_output_tensor_size[1]。 </td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>可选属性</td>
      <td><ul><li>表示输入`grad_output`在D、H和W维度上的空间大小。size为3，且各元素均大于零。必须满足：grad_output_tensor_size[2] == floor(input_size[2] * scales[0]) == output_size[0]；grad_output_tensor_size[3] == floor(input_size[3] * scales[1]) == output_size[1]；grad_output_tensor_size[4] == floor(input_size[4] * scales[2]) == output_size[2]。</li><li>默认值为{0, 0, 0}。</li></ul></td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td>
      <td><ul><li>指定沿每个维度的缩放数组，包含3个元素：scales_depth, scales_height, scales_width，对应公式中的`scales_d`、`scales_h`、`scales_w`。</li><li>默认值为{0.0f, 0.0f, 0.0f}。</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。数据类型和数据格式与入参`grad_output`的数据类型和数据格式保持一致。shape取决于输入`input_size`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest_exact3d_grad](examples/test_aclnn_upsample_nearest_exact3d_grad.cpp) | 通过[aclnnUpsampleNearestExact3dBackward](docs/aclnnUpsampleNearestExact3dBackward.md)接口方式调用UpsampleNearestExact3dGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/upsample_nearest_exact3d_grad_proto.h)构图方式调用UpsampleNearestExact3dGrad算子。         |

<!--[test_geir_upsample_nearest_exact3d_grad](examples/test_geir_upsample_nearest_exact3d_grad.cpp)-->
