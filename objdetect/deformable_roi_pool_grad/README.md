# DeformableRoiPoolGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：DeformableRoiPoolGrad 是 DeformableRoiPool（可变形感兴趣区域池化）的反向梯度算子，用于训练场景。它接收上游梯度 `grad`，结合前向输入 `x`（feature map）、`rois`、`offset`，计算对 `feature_map` 的梯度 `grad_x` 和对 `offset` 的梯度 `grad_offset`。

- 计算公式：

**grad_input（feature_map 梯度）**：将上游梯度按双线性插值权重反向分配到 4 个邻域点。

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

**grad_offset（offset 梯度）**（仅当 offset 不为空时计算）：

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
      <td>上游梯度，shape 为 (N, C, pH, pW)，N=ROI 数量，C=通道数，pH/pW=池化高宽。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>feature_map（前向输入），shape 为 (B, C, H, W)，B=batch，H/W=特征图高宽。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI 位置，shape 为 (N, 5)，格式 [batch_ind, x1, y1, x2, y2]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入（可选）</td>
      <td>偏移量，shape 为 (N, 2, pH, pW)，通道 0=w 方向，通道 1=h 方向；为空时不计算 grad_offset。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>ListInt，长度 2，[pooled_height, pooled_width]，池化输出尺寸。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>feature_map 相对原图的缩放比，默认 1.0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>属性</td>
      <td>采样比率，取值范围为 <code>[0, 46340]</code>；0 表示自适应 <code>ceil(roi_size/pooled_size)</code>，默认值为 0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>属性</td>
      <td>offset 缩放因子，默认 0.1。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>feature_map 梯度，shape 与 x 相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_offset</td>
      <td>输出</td>
      <td>offset 梯度，shape 与 offset 相同；offset 为空时输出零张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持 <term>Ascend 950PR/Ascend 950DT</term>、Atlas A2 训练/推理系列产品和 Atlas A3 训练/推理系列产品；Atlas 200I/500 A2 推理产品不在支持范围内。A2/A3 由 CANN 兼容交付中的既有 TBE 实现提供，本目录新增 SIMT Kernel 仅面向 Ascend 950。
- 所有张量注册格式均为 ND；文中 NCHW 仅描述逻辑轴顺序。
- 输入 `grad`、`x`、`rois`、非空 `offset` 及输出的数据类型必须一致，只支持 FLOAT。
- `grad` 和 `x` 必须为 4 维张量，`rois` 必须为 `(N, 5)`；要求 `grad.N == rois.N`、`grad.C == x.C`。
- 所有 Kernel 维度必须可由 INT32 表示，`x.H * x.W` 不得超过 INT32_MAX；`output_size` 必须恰有两个 `[1, INT32_MAX]` 范围内的元素，并与 `grad` 的 `pH/pW` 一致。
- 非空 `offset` 必须为 `(N, 2, pH, pW)`；无输入、0 维占位或 rank-1 的 `{0}` 占位均按空 `offset` 处理。
- `spatial_scale` 必须为有限正数；`sampling_ratio` 的支持范围为 `[0, 46340]`；`gamma` 必须为有限数。
- `rois` 格式为 `[batch_ind, x1, y1, x2, y2]`，合法输入的 `batch_ind` 应为 `[0, B)` 内的整数。Ascend 950 防御路径在 `B>0` 时将非有限 batch index 视为 0、将越界值夹到 `[0, B-1]`，小数按向零截断；ROI 坐标非有限时按 0 参与计算。`B=0` 时两个梯度输出均为零/空张量。
- 非空 `offset` 中的非有限值按 0 确定采样位置，对应方向的 `grad_offset` 写 0；该防御语义与正向 Ascend 950 实现一致。
- `sampling_ratio=0` 时，自适应采样数为 `ceil(roi_size/pooled_size)`；结果非有限、非正或单方向超过 46340 时，该 ROI 不产生有效采样。
- 采样坐标非有限或超出 `[-1, H] × [-1, W]` 时跳过该采样点。
- `grad_x` 使用单份 FLOAT workspace，用户 workspace 上限为 2 GiB；超过上限时 Tiling 返回失败。
- 输入必须连续（contiguous）。
- 采用确定性 owner 方案：每个输出位置由唯一线程按固定顺序累加，避免跨线程写冲突。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式 | [test_geir_deformable_roi_pool_grad](examples/test_geir_deformable_roi_pool_grad.cpp) | 通过[算子IR](op_graph/deformable_roi_pool_grad_proto.h)构图方式调用DeformableRoiPoolGrad算子。 |
