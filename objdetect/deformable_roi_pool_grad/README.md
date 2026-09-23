# DeformableRoiPoolGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：DeformableRoiPoolGrad是DeformableRoiPool（可变形感兴趣区域池化）的反向梯度算子，用于训练场景。它接收上游梯度`grad`，结合前向输入`x`（feature map）、`rois`、`offset`，计算对`feature_map`的梯度`grad_x`和对`offset`的梯度`grad_offset`。

- 计算公式：

**grad_input（feature_map梯度）**：将上游梯度按双线性插值权重反向分配到4个邻域点。

$$
\text{grad\_output\_this\_bin} = \text{grad\_output}[n,c,ph,pw] / \text{count}
$$

$$
\text{grad\_input}[\text{roi\_batch\_ind}, c, y_{low}, x_{low}] \mathrel{+}= \text{grad\_output\_this\_bin} \cdot w_1
$$

其中双线性插值权重：

$$
w_1 = (y_{high} - y) \cdot (x_{high} - x), \quad w_2 = (y_{high} - y) \cdot (x - x_{low})
$$

$$
w_3 = (y - y_{low}) \cdot (x_{high} - x), \quad w_4 = (y - y_{low}) \cdot (x - x_{low})
$$

**grad_offset（offset梯度）**（仅当offset不为空时计算）：

$$
\text{ogx} = \gamma \cdot \text{roi\_width} \cdot \text{grad\_output\_this\_bin} \cdot (\text{input}_{11} \cdot l_y + \text{input}_{10} \cdot (y_{high} - y_c) + \text{input}_{01} \cdot (-l_y) + \text{input}_{00} \cdot (y_c - y_{high}))
$$

$$
\text{ogy} = \gamma \cdot \text{roi\_height} \cdot \text{grad\_output\_this\_bin} \cdot (\text{input}_{11} \cdot l_x + \text{input}_{01} \cdot (x_{high} - x_c) + \text{input}_{10} \cdot (-l_x) + \text{input}_{00} \cdot (x_c - x_{high}))
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 280px">
  <col style="width: 150px">
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
      <td>上游梯度，shape为(N, C, pH, pW)，N=ROI数量，C=通道数，pH/pW=池化高宽。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>feature_map（前向输入），shape为(B, C, H, W)，B=batch，H/W=特征图高宽。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI位置，shape为(N, 5)，格式[batch_ind, x1, y1, x2, y2]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入（可选）</td>
      <td>偏移量，shape为(N, 2, pH, pW)，通道0=w方向，通道1=h方向；为空时不计算grad_offset。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>ListInt，长度2，[pooled_height, pooled_width]，池化输出尺寸。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>feature_map相对原图的缩放比，默认1.0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>属性</td>
      <td>采样比率，取值范围为<code>[0, 46340]</code>；0表示自适应<code>ceil(roi_size/pooled_size)</code>，默认值为0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>属性</td>
      <td>offset缩放因子，默认0.1。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>feature_map梯度，shape与x相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_offset</td>
      <td>输出</td>
      <td>offset梯度，shape与offset相同；offset为空时输出零张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持<term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>；<term>Atlas 200I/500 A2推理产品</term>不在支持范围内。<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>由CANN兼容交付中的既有TBE实现提供，本目录新增SIMT Kernel仅面向<term>Ascend 950PR&950DT系列产品</term>。
- 所有张量注册格式均为ND；文中NCHW仅描述逻辑轴顺序。
- 输入`grad`、`x`、`rois`、非空`offset`及输出的数据类型必须一致，只支持FLOAT。
- `grad`和`x`必须为4维张量，`rois`必须为`(N, 5)`；要求`grad.N==rois.N`、`grad.C==x.C`。
- 所有Kernel维度必须可由INT32表示，`x.H*x.W`不得超过INT32_MAX；`output_size`必须恰有两个`[1, INT32_MAX]`范围内的元素，并与`grad`的`pH/pW`一致。
- 非空`offset`必须为`(N, 2, pH, pW)`；无输入、0维占位或rank-1的`{0}`占位均按空`offset`处理。
- `spatial_scale`必须为有限正数；`sampling_ratio`的支持范围为`[0, 46340]`；`gamma`必须为有限数。
- `rois`格式为`[batch_ind, x1, y1, x2, y2]`，合法输入的`batch_ind`应为`[0, B)`内的整数。<term>Ascend 950PR&950DT系列产品</term>防御路径在`B>0`时将非有限batch index视为0、将越界值夹到`[0, B-1]`，小数按向零截断；ROI坐标非有限时按0参与计算。`B=0`时两个梯度输出均为零/空张量。
- 非空`offset`中的非有限值按0确定采样位置，对应方向的`grad_offset`写0；该防御语义与正向<term>Ascend 950PR&950DT系列产品</term>实现一致。
- `sampling_ratio=0`时，自适应采样数为`ceil(roi_size/pooled_size)`；结果非有限、非正或单方向超过46340时，该ROI不产生有效采样。
- 采样坐标非有限或超出`[-1, H] × [-1, W]`时跳过该采样点。
- `grad_x`使用单份FLOAT workspace，用户workspace上限为2GiB；超过上限时Tiling返回失败。
- 输入必须连续（contiguous）。
- 采用确定性owner方案：每个输出位置由唯一线程按固定顺序累加，避免跨线程写冲突。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式 | [test_geir_deformable_roi_pool_grad](examples/test_geir_deformable_roi_pool_grad.cpp) | 通过[算子IR](op_graph/deformable_roi_pool_grad_proto.h)构图方式调用DeformableRoiPoolGrad算子。 |
