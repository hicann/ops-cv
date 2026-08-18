# RoiAlignGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：ROIAlignGrad是ROIAlign算子的反向传播算子。ROIAlign是一种池化操作，用于从非均匀尺寸的特征图中提取固定尺寸的ROI（Region of Interest）特征。反向传播负责将输出梯度按正向传播时的双线性插值权重分配回输入特征图。

- 计算公式：

  对于每个ROI的每个bin中的每个采样点 $(y, x)$，通过双线性插值计算其4个邻居像素 $(y_{low}, x_{low})$、$(y_{low}, x_{high})$、$(y_{high}, x_{low})$、$(y_{high}, x_{high})$ 的插值权重：

  $$
  l_y = y - y_{low}, \quad l_x = x - x_{low}, \quad h_y = 1 - l_y, \quad h_x = 1 - l_x
  $$

  $$
  w_1 = h_y \cdot h_x, \quad w_2 = h_y \cdot l_x, \quad w_3 = l_y \cdot h_x, \quad w_4 = l_y \cdot l_x
  $$

  将每个采样点的梯度 $\frac{\partial L}{\partial y}$ 按权重累加到4个邻居像素：

  $$
  \text{xdiff}[b, c, y_{low}, x_{low}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_1
  $$

  $$
  \text{xdiff}[b, c, y_{low}, x_{high}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_2
  $$

  $$
  \text{xdiff}[b, c, y_{high}, x_{low}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_3
  $$

  $$
  \text{xdiff}[b, c, y_{high}, x_{high}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_4
  $$

  其中 $b$ 为batch索引，$c$ 为通道索引，$count = roi\_bin\_grid\_h \times roi\_bin\_grid\_w$ 为该bin的采样点总数。

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
      <td>ydiff</td>
      <td>输入</td>
      <td>反向传播梯度输入，shape为(N, C, pooled_height, pooled_width)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI坐标，shape为(N, 5)，每行格式为[batch_idx, roi_x1, roi_y1, roi_x2, roi_y2]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_n</td>
      <td>输入（可选）</td>
      <td>每个batch中有效ROI的数量，shape为(B,)。当前计算中未使用。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>xdiff_shape</td>
      <td>属性（必选）</td>
      <td>输出特征图的shape，即正向输入特征图的shape，格式为[B, C, H, W]，必须为4元素正整数列表。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>属性（必选）</td>
      <td>ROI输出特征图的高度。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>属性（必选）</td>
      <td>ROI输出特征图的宽度。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性（必选）</td>
      <td>特征图相对于原图的缩放比例。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>属性（可选）</td>
      <td>每个bin的采样次数。默认值为2，0表示自动计算（取ceil(roi_size / pooled_size)）。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>属性（可选）</td>
      <td>
        <ul style="margin-top: 0; margin-bottom: 0; padding-left: 2ch;">
          <li>0：无对齐。</li>
          <li>1：TF偏移模式（x2/y2 += spatial_scale，roi_w/roi_h取max(., 1.0)）。</li>
          <li>2：PyTorch对齐模式（x1/y1/x2/y2 -= 0.5）。</li>
          <li>3：MMDetection对齐模式（同2）。</li>
        </ul>
        默认值为1。
      </td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xdiff</td>
      <td>输出</td>
      <td>输入特征图的梯度，shape为(B, C, H, W)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持float32数据类型。
- ydiff和xdiff为4维ND格式，rois为2维ND格式。
- xdiff_shape属性必须为4元素正整数列表。
- 空tensor（shape含0维）时返回全零梯度。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn 接口调用 | [test_aclnn_roi_align_v2_backward_l2](examples/test_aclnn_roi_align_v2_backward.cpp) | 通过[aclnnRoiAlignV2Backward](./docs/aclnnRoiAlignV2Backward.md)接口方式调用RoiAlignGrad算子。 |
| 图模式      | [test_geir_roi_align_grad](examples/test_geir_roi_align_grad.cpp)                                    | 通过[算子IR](op_graph/roi_align_grad_proto.h)构图方式调用RoiAlignGrad算子。                         |
