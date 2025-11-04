# UpsampleBilinear2dAABackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[UpsampleBilinear2dAA](../upsample_bilinear2d_aa/README.md)的反向传播。

- 计算公式：对于一个二维插值点$(N, C, H, W)$, 插值$I(N, C, H, W)$可以表示为：
  
  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kW}\sum_{j=0}^{kH}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kW}w(i)/\sum_{j=0}^{kH}w(j)
  $$
  
  $$
  scaleH =\begin{cases}
  (inputSize[2]-1 / outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (inputSize[3]-1 / outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$
  
  - 其中：
    - $kW$、$kH$分别表示W方向和H方向影响插值点大小的点的数量
    - 如果$scaleH >= 1$，则$kH = floor(scaleH) * 2 + 1$，否则$kH = 3$
    - 如果$scaleW >= 1$，则$kW = floor(scaleW) * 2 + 1$，否则$kW = 3$
    - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
    - $w(i)$、$w(j)$是双线性抗锯齿插值的W方向和H方向权重，计算公式为：
      
      $$
        w(i) = \begin{cases}
        1 - |h_i - h| & |h_i -h| < 1 \\
        0 & otherwise
        \end{cases}
      $$
      
      $$
        w(j) = \begin{cases}
        1 - |w_j - w| & |w_j -w| < 1 \\
        0 & otherwise
        \end{cases}
      $$

  - 假设：正向插值的输出图像out $(h, w)$受原图像input $(h_i, w_j)$影响，则有:
  
    $$
    gradInput(h_i,w_j) += gradOutput(h,w) * f(h_i,w_j)
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
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定输出空间大小，对应公式中的`outputSize`。size需要等于2，且各元素均大于0。表示指定`grad_output`在H和W维度上的空间大小。</li><li>默认值为空。</li></ul></td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>指定输出空间大小，对应公式中的`inputSize`。size为4，且各元素均大于零。表示输出`grad_input`分别在N、C、H和W维度上的空间大小。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则输入和输出张量的左上角顶点及两条边对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为空。</li></ul></td>
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
| aclnn接口  | [test_aclnn_upsample_bilinear2d_aa_backward](examples/test_aclnn_upsample_bilinear2d_aa_backward.cpp) | 通过[aclnnUpsampleBilinear2dAABackward](docs/aclnnUpsampleBilinear2dAABackward.md)接口方式调用UpsampleBilinear2dAABackward算子。 |
<!--
| 图模式 | [test_geir_upsample_bilinear2d_aa_backward](examples/test_geir_upsample_bilinear2d_aa_backward.cpp)  | 通过[算子IR](op_graph/upsample_bilinear2d_aa_backward_proto.h)构图方式调用UpsampleBilinear2dAABackward算子。         |
-->