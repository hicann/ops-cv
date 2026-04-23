# RoiAlignV2

## 产品支持情况
| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：对输入特征图执行 ROI Align 操作，对每个感兴趣区域（ROI）进行双线性插值采样，输出固定大小的特征图。
- 输入 ROI 坐标格式为 <batch_index, x1, y1, x2, y2>（左上角和右下角坐标），在代码中以一维数组形式存储（[numRois, 5] 维度展开为长度 numRois*5 的一维数据）。
- 计算公式：
  - 输入节点：
    - features (shape[N, C, H, W], FLOAT32) - 输入特征图
    - rois (shape[numRois, 5], FLOAT32) - 感兴趣区域坐标（batch_index, x1, y1, x2, y2）

  - 计算节点：
    - Step1: 将 ROI 坐标乘以 spatial_scale 进行缩放，并转换为 (x, y, w, h) 格式；
    - Step2: 根据 pooled_height 和 pooled_width 将 ROI 区域划分为均匀的 bin，计算每个 bin 的宽高 (bin_w, bin_h)；
    - Step3: 根据 sampling_ratio 确定每个 bin 内的采样网格大小 (grid_h, grid_w)，若 sampling_ratio > 0 则固定为该值，否则自适应计算 (ceil(roi_h / pooled_height), ceil(roi_w / pooled_width))；
    - Step4: 对每个 bin 内的每个采样点，计算其在特征图上的坐标，通过双线性插值获取特征值；
    - Step5: 对每个 bin 内所有采样点的特征值取平均，作为该位置的输出值；
    - 重复上述步骤直至所有 ROI 的所有通道处理完成。

  - 输出节点：
    - output (shape[numRois, C, pooled_height, pooled_width], FLOAT32) - 对齐后的 ROI 特征图

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
      <td>features</td>
      <td>输入</td>
      <td>输入特征图，shape [N, C, H, W]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>感兴趣区域坐标，shape [numRois, 5]，每行为 (batch_index, x1, y1, x2, y2)。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>对齐后的 ROI 特征图，shape [numRois, C, pooled_height, pooled_width]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>属性（可选）</td>
      <td>输出特征图的高度。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>属性（可选）</td>
      <td>输出特征图的宽度。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性（可选）</td>
      <td>空间缩放因子，用于将 ROI 坐标映射到特征图尺度。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>属性（可选）</td>
      <td>每个 bin 的采样点数。大于 0 时固定为该值，否则自适应计算。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 目前只支持 float32 输入
- 目前只支持 ascend910b

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_roi_align_v2](./examples/test_aclnn_roi_align_v2.cpp) | 通过 aclnnRoiAlignV2 接口方式调用 RoiAlignV2 算子。 |
