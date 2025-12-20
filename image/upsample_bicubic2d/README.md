# UpsampleBicubic2d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用2D双三次上采样。如果输入Tensor x的shape为(N, C, H, W)，则输出Tensor out的shape为(N, C, outputSize[0], outputSize[1])。
- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$out(N, C, h, w)$可以表示为：
  
  $$
  {out(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$
  
  $$
  scaleH =\begin{cases}
  (self.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  self.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (self.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  self.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$
  
  其中：
  - i和j是$W(i, j)$的索引变量。
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值。
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
      <td>input</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td><!--aclnn多增了一个NCHW-->
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>指定输出空间大小，对应公式中的`outputSize`。size需要等于2，且各元素均大于0。表示指定`output`在H和W维度上的空间大小。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>INT64</td>
      <td>-</td>
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
      <td>output</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`out`。数据类型和数据格式需要与`self`的数据类型和数据格式一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：输入和输出的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bicubic2d](examples/test_aclnn_upsample_bicubic2d.cpp) | 通过[aclnnUpsampleBicubic2d](docs/aclnnUpsampleBicubic2d.md)接口方式调用UpsampleBicubic2d算子。 |
<!--
| 图模式 | [test_geir_upsample_bicubic2d](examples/test_geir_upsample_bicubic2d.cpp)  | 通过[算子IR](op_graph/upsample_bicubic2d_proto.h)构图方式调用UpsampleBicubic2d算子。         |
-->