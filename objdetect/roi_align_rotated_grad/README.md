# RoiAlignRotatedGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Atlas A3 训练系列产品/Atlas A3 推理系列产品   |     √    |
|  Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件     |     √    |

## 功能说明

- 算子功能：通过旋转框各点坐标将梯度回传至对应位置。

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
      <td>x_grad</td>
      <td>输入</td>
      <td>输入特征图的梯度。</td>
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
      <td>y_grad_shape</td>
      <td>属性</td>
      <td>输出的shape。</td>
      <td>LISTINT</td>
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
      <td>是否量化。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clockwise</td>
      <td>属性</td>
      <td>时钟方向。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>输出</td>
      <td>输出特征图。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* 输入x_grad的shape必须为[N,C,pooled_h,pooled_w]的Tensor。C的取值在[0, 1024]之间，pooled_h、pooled_w的取值在[0, 1024]之间。
* Rois的shape必须为[n, 6]的Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_roi_align_rotated_grad](examples/test_aclnn_roi_align_rotated_grad.cpp) | 通过[aclnnRoiAlignRotatedGrad](docs/aclnnRoiAlignRotatedGrad.md)接口方式调用RoiAlignRotatedGrad算子。 |