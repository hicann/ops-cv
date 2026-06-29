# AnchorResponseFlags

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：在目标检测网络中生成锚框（anchor）的响应标志。根据真值框（ground truth bounding boxes）的中心点位置，确定哪些锚框网格位置负责检测目标，并生成对应的标志位。

- 计算公式：

$$
cx_i = (x1_i + x2_i) \times 0.5
$$

$$
cy_i = (y1_i + y2_i) \times 0.5
$$

$$
grid\_x_i = \lfloor cx_i / stride\_h \rfloor
$$

$$
grid\_y_i = \lfloor cy_i / stride\_w \rfloor
$$

$$
grid\_idx_i = grid\_y_i \times feat\_w + grid\_x_i
$$

$$
responsible\_grid[grid\_idx_i] = 1
$$

$$
output = repeat(responsible\_grid, num\_base\_anchors)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>gt_bboxes</td>
      <td>输入</td>
      <td>真值框坐标，shape为[N, 4]，格式为[x1, y1, x2, y2]。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>flags</td>
      <td>输出</td>
      <td>锚框响应标志，shape为[feat_h * feat_w * num_base_anchors]，值为0或1。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>featmap_size</td>
      <td>属性</td>
      <td>特征图大小，长度为2的列表[feat_h, feat_w]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>步长，长度为2的列表[stride_h, stride_w]，值必须为正整数。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_base_anchors</td>
      <td>属性</td>
      <td>每个网格位置的锚框数量，正整数。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 gt_bboxes 必须为 2D tensor，第二维必须为 4。
2. featmap_size 和 strides 必须为长度为 2 的列表。
3. strides 中的值必须为正整数（不包含 0）。
4. num_base_anchors 必须为正整数。
5. 输出 dtype 固定为 uint8，不随输入 dtype 变化。
6. 当 gt_bboxes 为空（N=0）时，输出全零 tensor。
7. 中心点映射后超出特征图范围时，grid 索引会被裁剪到有效范围内。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_anchor_response_flags.cpp">test_geir_anchor_response_flags</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
