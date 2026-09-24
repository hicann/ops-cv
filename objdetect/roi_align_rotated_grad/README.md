# RoiAlignRotatedGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     ×    |

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
      <td>是否量化，true则将目标框中心坐标值减0.5。</td>
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
      <td>y_grad</td>
      <td>输出</td>
      <td>输出特征图。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x_grad为4维(N, pooled_h, pooled_w, C)，通道数C取值范围[0, 1024]，pooled_h、pooled_w取值范围[0, 1024]。
- 输入rois为2维(6, N)，第0维固定为6（字段数），第1维N为ROI数量且必须等于x_grad.dim(0)。
- x_grad.dim(1)必须等于属性pooled_h，x_grad.dim(2)必须等于属性pooled_w。
- x_grad.dim(3)必须等于y_grad_shape[3]（通道数一致）。
- 属性y_grad_shape长度必须为4，各元素为非负整数，语义为[B, H, W, C]。
- 属性pooled_h、pooled_w必须大于0；sampling_ratio必须大于等于0。
- rois第0行batch_ind必须为整数值且落在[0, y_grad_shape[0])内，kernel按截断取整（向零取整）解释该值，越界属未定义行为；第5行angle为弧度值。
- 输出y_grad为梯度累加值，可正可负，无值域裁剪；未被ROI覆盖的位置为0。
- 算子为精度敏感算子，禁止框架降精度处理，仅支持float32单一数据类型组合。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_roi_align_rotated_grad](examples/test_geir_roi_align_rotated_grad.cpp) | 通过图模式方式调用RoiAlignRotatedGrad算子。 |
