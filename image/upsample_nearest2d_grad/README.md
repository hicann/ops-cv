# UpsampleNearest2dGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[UpsampleNearest](../upsample_nearest/README.md)在exact_mode为false时的反向传播。
- 计算公式：

  $$
  gradInput(N, C, H, W) += gradOutput( N, C, ceil ( scales\_h * H ),  ceil ( scales\_w * W )) 
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
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOutput`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td><!--aclnn多增了一个NCHW-->
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>表示指定`grad_output`在H和W维度上的空间大小。size需要等于2，且各元素均大于0。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>表示指定`grad_input`在H和W维度上的空间大小。size为4，且最后两个元素均大于零。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认(ize大小为4，且最后两个元素均大于零。当输入gradOut的数据格式为NCHW时，表示输出gradInput分别在N、C、H和W维度上的空间大小；当输入gradOut的数据格式为NHWC时，表示输出gradInput分别在N、H、W和C维度上的空间大小。)-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>表示输出`grad_input`的height维度乘数，对应公式中的`scales_h`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>表示输出`grad_input`的width维度乘数，对应公式中的`scales_w`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。数据类型和数据格式与入参`grad_output`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest1d_backward](examples/test_aclnn_upsample_nearest1d_backward.cpp) | 通过[aclnnUpsampleNearest1dBackward](docs/aclnnUpsampleNearest1dBackward.md)接口方式调用UpsampleNearest2dGrad算子。 |
| aclnn接口  | [test_aclnn_upsample_nearest2d_grad](examples/test_aclnn_upsample_nearest2d_grad.cpp) | 通过[aclnnUpsampleNearest2dBackward](docs/aclnnUpsampleNearest2dBackward.md)接口方式调用UpsampleNearest2dGrad算子。 |
<!--
| 图模式 | [test_geir_upsample_nearest_exact2d_grad](examples/test_geir_upsample_nearest_exact2d_grad.cpp)  | 通过[算子IR](op_graph/upsample_nearest_exact2d_grad_proto.h)构图方式调用UpsampleNearest2dGrad算子。         |
-->