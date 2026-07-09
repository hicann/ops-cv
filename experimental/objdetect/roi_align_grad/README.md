# ROIAlignGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：ROIAlign 的反向传播。ROIAlign 是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图；本算子将正向输出的梯度 `y_diff` 依据感兴趣区域 `rois` 反向散射累加回输入特征图梯度 `x_diff`。
- 计算公式：
  - 输入节点：
    - y_diff (shape[K,C,pooled_height,pooled_width], FLOAT32) - 反向传播的输入梯度，K 为 roi 个数。
    - rois (shape[K,5], FLOAT32) - 感兴趣区域坐标 (image_id, x1, y1, x2, y2)。
  - 计算节点：
    - Step1: 依据 `spatial_scale` 将 roi 坐标映射到输入特征图尺度；`roi_end_mode` 为 2 时坐标偏移 -0.5 使相邻像素索引更好对齐。
    - Step2: 对每个输出网格 (pooled_height × pooled_width)，按 `sample_num` 在 bin 内均匀采样，计算双线性插值的 4 个邻点坐标与权重。
    - Step3: 将 `y_diff` 上对应位置的梯度按双线性权重散射累加（原子写回）到 `x_diff` 的 4 个邻点。
  - 输出节点：
    - x_diff (shape 由 `xdiff_shape` 指定 [B,C,inputHeight,inputWidth], FLOAT32) - 反向传播的输出梯度。

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
      <td>y_diff</td>
      <td>输入</td>
      <td>反向传播的输入梯度 [K,C,pooled_height,pooled_width]。</td>
      <td>FLOAT32</td>
      <td>ND/NCHW/NC1HWC0</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>感兴趣区域坐标 [K,5]，5 代表 (image_id, x1, y1, x2, y2)。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_diff</td>
      <td>输出</td>
      <td>反向传播的输出梯度 [B,C,inputHeight,inputWidth]。</td>
      <td>FLOAT32</td>
      <td>ND/NCHW/NC1HWC0</td>
    </tr>
    <tr>
      <td>xdiff_shape</td>
      <td>属性</td>
      <td>正向输入的 shape，用来指定反向输出 x_diff 的 shape (B,C,inputHeight,inputWidth)。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>属性</td>
      <td>正向 ROIAlign 池化后输出图像的宽度。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>属性</td>
      <td>正向 ROIAlign 池化后输出图像的高度。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>乘法空间尺度因子，将 roi 坐标从输入空间尺度转换为池化时使用的尺度，需大于 0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>属性（可选，默认 2）</td>
      <td>ROIAlign 中每个输出元素在 H 和 W 方向上的采样频率，需大于等于 0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>属性（可选，默认 1）</td>
      <td>roi 坐标对齐模式，2 时坐标偏移 -0.5 对齐相邻像素索引。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 目前只支持 float32 输入。
- rois 第 1 维固定为 5，且第 0 维需与 y_diff 第 0 维（K）保持一致。
- image_id 取值范围 [0, B)，B 为 xdiff_shape 的第一个值。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_roi_align_grad](./examples/test_aclnn_roi_align_grad.cpp) | 通过 aclnnRoiAlignV2Backward 接口方式调用 ROIAlignGrad 算子。 |

## 贡献说明

| 贡献方 | 贡献者 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- |
| 西北工业大学智能感知交互实验室 | Xzz | 2026/7/8 | ROIAlignGrad 算子适配开源仓 |
