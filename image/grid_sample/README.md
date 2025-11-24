# GridSample

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
- 计算公式：

  - 计算流程：
    1. 根据grid存储的(x, y)或者(x, y, z)值，计算出映射到input上坐标，坐标和alignCorners、paddingMode有关。
    2. 坐标根据输入的interpolationMode，选择使用bilinear、nearest、bicubic不同插值模式计算输出值。
  
  - 其中：
    - 3D场景：
  
      输入input、grid网格、输出output的尺寸如下：
  
      $$
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      output: (N, C, D_{out}, H_{out}, W_{out})
      $$
  
      其中input、grid、output中的N是一致的，input和output中的C是一致的，grid和output中的$D_{out}$、$H_{out}$、$W_{out}$是一致的，grid最后一维大小为3，表示input像素位置信息为(x, y, z)，一般会将x、y、z的取值范围归一化到[-1, 1]之间。
  
    - 2D场景：
      
      输入input、grid网格、输出output的尺寸如下：
    
      $$
      input: (N, C, H_{in}, W_{in})\\
      grid: (N, H_{out}, W_{out}, 2)\\
      output: (N, C, H_{out}, W_{out})
      $$
  
      其中input、grid、output中的N是一致的，input和output中的C是一致的，grid和output中的$H_{out}$、$W_{out}$是一致的，grid最后一维大小为2，表示input像素位置信息为(x, y)，一般会将x和y的取值范围归一化到[-1, 1]之间，(-1, 1)表示左上角坐标，(1, -1)表示右下角坐标。
    
    
    - 对于超出范围的坐标，会根据paddingMode进行不同处理：
  
      - paddingMode="zeros"，表示对越界位置用0填充。
      - paddingMode="border"，表示对越界位置用边界值填充。
      - paddingMode="reflection"，表示对越界位置用边界值的对称值填充。
  
    - 对input采样时，会根据interpolationMode进行不同处理：
  
      - interpolationMode="bilinear"，表示取input中(x, y)或者(x, y, z)周围四个坐标的加权平均值。
      - interpolationMode="nearest"，表示取input中距离(x, y)或者(x, y, z)最近的坐标值。
      - interpolationMode="bicubic"，表示取input中(x, y)或者(x, y, z)周围十六个坐标的加权平均值。

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
      <td>进行插值计算的输入张量，对应公式中描述的`input`。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>采样的网络，对应公式中描述的`grid`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td><!--相对IR原型新增了BF16-->
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示插值模式，对应公式描述中的`interpolationMode`。支持bilinear（双线性插值）、nearest（最邻近插值）、bicubic（双三次插值）。</li><li>默认值为"bilinear"。</li></ul></td><!--aclnn是三种 0：bilinear（双线性插值），1：nearest（最邻近插值），2：bicubic（双三次插值） -->
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示填充模式，对应公式描述中的`paddingMode`。支持zeros(0)、border(1)、reflection(2)三种模式。</li><li>默认值为"zeros"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>channel_last</td>
      <td>可选属性</td>
      <td><ul><li>指定输入x的C轴是否在最后。如果为true，则x的形状为`[batch, height, width, channels]`，否则为`[batch, channels, height, width]`。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scheduler_mode</td>
      <td>可选属性</td>
      <td><ul><li>调度模式，执行的操作方式。0：一般；1：滑动窗口。仅当channel_last的值为true时，配置为1有效。</li><li>默认值为1。</li></ul></td><!--IR中接口中默认值是1，描述中默认值是0，aclnn没有这个参数-->
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>插值计算的最终输出结果，对应公式中描述的`output`。
      </td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
  </tbody></table>


- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：2D场景下，数据类型不支持BFLOAT16。

## 约束说明

- 3D场景下
  - 属性`interpolation_mode`不支持配置为"bicubic"。
  - 输入`x`的（D轴的大小 * H轴的大小 * W轴的大小） < INT32的最大值。
- 2D场景下，输入`x`的（H轴的大小 * W轴的大小） < INT32的最大值。
- `x`和`grid`的shape，所有维度都必须大于0。
- 输入`x`的（D轴的大小 * H轴的大小 * W轴的大小） < INT32的最大值。
- grid的输入值*图片（长或宽）大于24位的二进制数（16777216），采样点可能存在误差，精度可能产生偏差。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：如果grid存在超出[-1, 1]范围的数据，使用bicubic插值时，小值域数据计算可能存在误差，精度可能产生偏差。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grid_sample2_d](examples/test_aclnn_grid_sample2_d.cpp) | 通过[aclnnGridSampler2D](docs/aclnnGridSampler2D.md)接口方式调用GridSample算子。 |
| aclnn接口  | [test_aclnn_grid_sample3_d](examples/test_aclnn_grid_sample3_d.cpp) | 通过[aclnnGridSampler3D](docs/aclnnGridSampler3D.md)接口方式调用GridSample算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grid_sample_proto.h)构图方式调用GridSample算子。         |

<!--[test_geir_grid_sample](examples/test_geir_grid_sample.cpp)-->