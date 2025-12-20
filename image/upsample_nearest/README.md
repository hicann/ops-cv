# UpsampleNearest

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：
 
  对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
  - 如果输入shape为（N，C，L），则输出shape为（N，C，outputSize）；
  - 如果输入shape为（N，C，H，W），则输出shape为（N，C，outputSize[0]，outputSize[1]）。

- 计算公式：

  - 当exact_mode=true时：

    $$
    h_{src} = min(floor((h_{dst} + 0.5) * scalesH),  H - 1)
    $$

    $$
    w_{src} = min(floor((w_{dst} + 0.5) * scalesW),  W - 1)
    $$

    $$
    out(N, C, h_{dst}, w_{dst}) = self(N, C, h_{src}, w_{src})
    $$

  - 当exact_mode=false时：

    $$
    h_{src} = min(floor(h_{dst} * scalesH),  H - 1)
    $$

    $$
    w_{src} = min(floor(w_{dst} * scalesW),  W - 1)
    $$

    $$
    out(N, C, h_{dst}, w_{dst}) = self(N, C, h_{src}, w_{src})
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
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td><!--aclnn多增了一个NCHW-->
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>指定输出空间大小，对应公式中的`outputSize`。size需要等于2，表示指定`y`在H和W维度上的空间大小。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exact_mode</td>
      <td>可选属性</td><!--aclnn没有这个参数-->
      <td><ul><li>是否使用exact模式，对应公式描述中的`exact_mode`。</li><li>默认值为false。</li></ul></td><!--公式是否体现-->
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`out`。数据类型与入参`x`的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas 推理系列产品</term>：输入和输出的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest](examples/test_aclnn_upsample_nearest.cpp) | 通过[aclnnUpsampleNearestExact1d](docs/aclnnUpsampleNearestExact1d.md)接口方式调用UpsampleNearest算子。 |
| aclnn接口  | [test_aclnn_upsample_nearest_exact2d](examples/test_aclnn_upsample_nearest_exact2d.cpp) | 通过[aclnnUpsampleNearestExact2d](docs/aclnnUpsampleNearestExact2d.md)接口方式调用UpsampleNearest算子。 |
<!--
| 图模式 | [test_geir_upsample_nearest_exact2d](examples/test_geir_upsample_nearest_exact2d.cpp)  | 通过[算子IR](op_graph/upsample_nearest_exact2d_proto.h)构图方式调用UpsampleNearest算子。         |
-->