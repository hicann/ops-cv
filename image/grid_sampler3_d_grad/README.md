# GridSampler3DGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：[GridSampler](../grid_sample/README.md)中3D场景的反向传播，完成张量input与张量grid的梯度计算。

- 计算公式：

  - 计算流程：
    1. 根据grid存储的(x, y, z)值，计算出映射到input上的坐标，坐标和align_corners、padding_mode有关。
    2. 坐标根据输入的interpolation_mode，选择使用bilinear、nearest、bicubic不同插值模式计算输出值。
    3. 根据grad存储的梯度值乘上对应点的权重值，计算出最终dx、dgrid的结果。
  
  - 其中：
    - 3D场景：
  
      grad、input、grid、dx、dgrid的尺寸如下：
  
      $$
      grad: (N, C, D_{in}, H_{in}, W_{in})\\
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      dx: (N, C, D_{in}, H_{in}, W_{in})\\
      dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
      $$
  
      其中grad、input、grid、dx、dgrid中的N是一致的，grad、input和dx中的C是一致的，grad、input和dx中的$D_{in}$、$H_{in}$、$W_{in}$是一致的，grid和dgrid中的$D_{out}$、$H_{out}$、$W_{out}$是一致的，grid最后一维大小为3，表示x像素位置信息为(x, y, z)，一般会将x、y、z的取值范围归一化到[-1, 1]之间。
   
    
    - 对于超出范围的坐标，会根据padding_mode进行不同处理：
  
      - padding_mode="zeros"，表示对越界位置用0填充。
      - padding_mode="border"，表示对越界位置用边界值填充。
      - padding_mode="reflection"，表示对越界位置用边界值的对称值填充。
  
    - 对input采样时，会根据interpolation_mode进行不同处理：
  
      - interpolation_mode="bilinear"，表示取input中(x, y, z)周围四个坐标的加权平均值。
      - interpolation_mode="nearest"，表示取input中距离(x, y, z)最近的坐标值。


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
      <td>grad</td>
      <td>输入</td>
      <td>表示反向传播过程中上一层的输出梯度，对应公式描述中的`grad`。数据类型与`x`的数据类型一致。当数据类型DOUBLE时，数据格式不支持NDHWC。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td><!--IR原型仅有NCDHW，补充了aclnn的NDHWC，补充了NDHWC的同样原理-->
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示反向传播的输入张量，对应公式描述中的`input`。shape仅支持五维，且需满足`x`和`grad`的shape保持一致，x的D，H，W值不可为0。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td><!--IR原型仅有NCDHW，补充了aclnn的NDHWC，补充了NDHWC的同样原理-->
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>表示采用像素位置的张量，对应公式描述中的`grid`。C轴的值必须为3，shape仅支持五维，且需满足`grid`和`grad`的N轴的值保持一致。</td><!--IR中最后一维是2？-->
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td><!--参考aclnn新增了double类型-->
      <td>ND</td><!--参考aclnn的数据格式是：NDHWC，这个怎么写？，IR原型中是NDHW3-->
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示插值模式，对应公式描述中的`interpolation_mode`。支持bilinear（0：双线性插值）和nearest（1：最邻近插值）。</li><li>默认值为"bilinear"。</li></ul></td>
      <td>STRING</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td><ul><li>用于表示填充模式，对应公式描述中的`padding_mode`。支持0（'zeros'） 、 1（'border'） 、2（'reflection'）三种模式。</li><li>默认值为"zeros"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>表示设定特征图坐标与特征值的对应方式，对应公式描述中的`align_corners`。如果为true，则将极值-1和1视为参考输入的角像素点的中心点；如果为false，则视为参考输入的角像素点的角点。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应公式描述中的`dx`。数据类型、数据格式和shape与`x`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td><!--参考aclnn新增了double类型-->
      <td>NCDHW、NDHWC</td><!--IR原型仅有NCDHW，补充了aclnn的NDHWC，补充了NDHWC的同样原理-->
    </tr>
    <tr>
      <td>dgrid</td>
      <td>输出</td>
      <td>表示`grid`梯度，对应公式描述中的`dgrid`。数据类型、数据格式和shape与`grid`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td><!--参考aclnn新增了double类型-->
      <td>NCDHW、NDHWC</td><!--IR原型仅有NCDHW，补充了aclnn的NDHWC，补充了NDHWC的同样原理-->
    </tr>
  </tbody></table>

## 约束说明

无
<!--
GridSampler3DGrad默认为非确定性实现，暂不支持确定性实现，[确定性计算](./docs/context/确定性计算.md)配置后不会生效。
-->

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grid_sampler3_d_backward](examples/test_aclnn_grid_sampler3_d_backward.cpp) | 通过[aclnnGridSampler3DBackward](docs/aclnnGridSampler3DBackward.md)接口方式调用GridSampler3DGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grid_sampler3_d_grad_proto.h)构图方式调用GridSampler3DGrad算子。         |

<!--[test_geir_grid_sampler3_d_grad](examples/test_geir_grid_sampler3_d_grad.cpp)-->