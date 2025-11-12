# RoiAlignRotated

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Atlas A3 训练系列产品/Atlas A3 推理系列产品   |     √    |
|  Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件     |     √    |

## 功能说明

- 算子功能：用于旋转候选框的ROI对齐池化层。

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
      <td>输入特征图。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI边界框。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>ROI边界框的缩放率。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>属性</td>
      <td>计算RoiAlign时的采样率。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>属性</td>
      <td>RoiAlign输出时池化特征图的高。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>属性</td>
      <td>RoiAlign输出时池化特征图的宽。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>属性</td>
      <td>是否量化，true则将目标框中心坐标值-0.5。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clockwise</td>
      <td>属性</td>
      <td>时钟方向，true为顺时针。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>输出特征图。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* 输入shape必须为[N,C,W,H]的Tensor，其中N、C、W和H均为INT32类型正整数。C的取值在[0, 1024]之间。
* rois的shape必须为[n, 6]的Tensor。n的取值范围为[1, 8192]。对于每个ROI而言，其组成为[batch_idx, center_x, center_y, w, h, angle]，其中batch_idx取值范围在[0, N)之间，会进行强制类型转换(float->int)，center_x、center_y、w、h为FLOAT32类型正浮点数，center_x与w的取值范围为[0, W)，center_y与h的取值范围为[0, H)，angle的取值为FLOAT32类型浮点数，取值范围为[0, π)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_roi_align_rotated](examples/test_aclnn_roi_align_rotated.cpp) | 通过[aclnnRoiAlignRotated](docs/aclnnRoiAlignRotated.md)接口方式调用RoiAlignRotated算子。 |