# UpsampleLinear1d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用线性插值算法进行上采样。如果输入shape为（N，C，L），则输出shape为（N，C，outputSize）。

- 计算公式：
  - 核心算法逻辑：
    1.将目标图像缩放到和原始图像一样大的尺寸。
    2.计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3.分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时有不同。则有以下公式：

    $$
    scale =\begin{cases}
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales & alignCorners=false\&scales>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$
   
    那么，对于output的某个方向上面的点p(x)，映射回原始图像中的点记为q(x')，则有关系: 
    
    $$
    x' =\begin{cases}
    x * scale\_h & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$
    
    - 记：
    
      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$
   
    - 则有以下公式：
      $$
      {V(p_{x})} = {V(p_{x0})} * {lambda_{0}}  + {V(p_{x1})} * {lambda_{1}} 
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
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td><!--aclnn是必选输入-->
      <td>指定输出空间大小，对应公式中的`outputSize`。size大小为1，且取值大于0。表示指定`y`在H和W维度上的空间大小。</td><!--opdef中是否是2维不确定，这个参考的是aclnn，待确认-->
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。如果设置为true，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值；如果设置为false，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于外界边值。</li><li>默认值为false。</li></ul></td><!--并且插值使用边缘值填充用于外界边值，使此操作在保持不变时独立于输入大小`scales`。(来源于aclnn约束）-->
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>表示输出`y`的L维度乘数，对应公式中的`scales`。</li><li>无默认值。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式描述中的`output`。数据类型、数据格式与入参`x`的数据类型、数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_linear1d](examples/test_aclnn_upsample_linear1d.cpp) | 通过[aclnnUpsampleLinear1d](docs/aclnnUpsampleLinear1d.md)接口方式调用UpsampleLinear1d算子。 |
<!--
| 图模式 | [test_geir_upsample_linear1d](examples/test_geir_upsample_linear1d.cpp)  | 通过[算子IR](op_graph/upsample_linear1d_proto.h)构图方式调用UpsampleLinear1d算子。         |
-->