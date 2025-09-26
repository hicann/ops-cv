# UpsampleBicubic2dGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[UpsampleBicubic2d](../upsample_bicubic2d/README.md)的反向传播。如果输入张量grad_output的shape为(N, C, H, W)，则输出张量grad_input的shape为(N, C, inputSize[2], inputSize[3])。

- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$gradInput(N, C, h, w)$可以表示为：
  
  $$
  {gradInput(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$
  
  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$
  
  其中：
  - $f(h_i, w_j)$是gradOutput在$(h_i, w_j)$的像素值。
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：
    
    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$
    
    其中：
    - $a=-0.75$
    - $d = |(h, w) - (h_i, w_j)|$


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
      <td>表示反向计算的梯度Tensor，对应公式描述中的`gradOutput`。数据类型与输出`grad_input`的数据类型一致。</td><!--aclnn有仅支持4维的描述，待确认是否体现。-->
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td><!--aclnn多增了一个NCHW-->
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>表示输出`grad_input`的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为0.0。</li></ul></td><!--aclnn有次约束，看看是否合入：不能传入负值。-->
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>表示输出`grad_input`的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为0.0。</li></ul></td><!--aclnn有次约束，看看是否合入：不能传入负值。-->
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`gradInput`。数据类型与输入`grad_output`的数据类型一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bicubic2d_grad](examples/test_aclnn_upsample_bicubic2d_grad.cpp) | 通过[aclnnUpsampleBicubic2dBackward](docs/aclnnUpsampleBicubic2dBackward.md)接口方式调用UpsampleBicubic2dGrad算子。 |
<!--
| 图模式 | [test_geir_upsample_bicubic2d_backward](examples/test_geir_upsample_bicubic2d_backward.cpp)  | 通过[算子IR](op_graph/upsample_bicubic2d_backward_proto.h)构图方式调用UpsampleBicubic2dGrad算子。         |
-->