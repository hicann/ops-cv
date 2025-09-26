# UpsampleNearestExact3d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|

|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1.将目标图像缩放到和原始图像一样大的尺寸。
    2.对于缩放之后的目标图像的点，计算距离最近的原始图像的点，后者的值直接复制给前者。
  - 具体计算逻辑：
    则计算缩放系数有以下公式：
    
    $$
    scale = self.dim / outputSize
    $$
    
    那么，对于out的某个方向上面的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系: 
    
    $$
    x' = \min(\lfloor (x+0.5) * scale \rfloor, self.dim(2) - 1)
    $$
    $$
    y' = \min(\lfloor (y+0.5) * scale \rfloor, self.dim(3) - 1)
    $$
    $$
    z' = \min(\lfloor (z+0.5) * scale \rfloor, self.dim(4) - 1)
    $$
    
    则有以下公式：
    
    $$
    {V(p_{x,y,z})} = {V(q_{x',y',z'})}
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
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>指定输出`out`的Tensor大小，对应公式中的`outputSize`。size大小为3，各元素均大于零。表示输出`out`在D、H和W维度上的空间大小。</td><!--opdef中是否是3维不确定，这个参考的是aclnn，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定沿每个维度的缩放数组，包含3个元素：scale_depth, scale_height, scale_width，对应公式中的`scale`。</li><li>默认值为[0.0, 0.0, 0.0]。</li></ul></td><!--算子侧不包含只能指定'scales'和'output_size'中的一个。如果两者都指定则会产生错误。-->
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式描述中的`out`。数据类型和数据格式需与入参`self`的数据类型和数据格式保持一致，shape取决于输入x和output_size/scales。</td><!--最后一句跟scales的参数说明中的二选一是不是有些矛盾啊-->
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

<term>Atlas 推理系列产品</term>：输入和输出的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest_exact3d](examples/test_aclnn_upsample_nearest_exact3d.cpp) | 通过[aclnnUpsampleNearestExact3d](docs/aclnnUpsampleNearestExact3d.md)接口方式调用UpsampleNearestExact3d算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/upsample_nearest_exact3d_proto.h)构图方式调用2UpsampleNearestExact3d算子。         |

<!--[test_geir_upsample_nearest_exact3d](examples/test_geir_upsample_nearest_exact3d.cpp)-->