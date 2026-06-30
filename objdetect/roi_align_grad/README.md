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

ROIAlignGrad是ROIAlign算子的反向传播算子。ROIAlign是一种池化操作，用于从非均匀尺寸的特征图中提取固定尺寸的ROI（Region of Interest）特征。反向传播负责将输出梯度按正向传播时的双线性插值权重分配回输入特征图。

## 约束说明

- 仅支持float32数据类型。
- ydiff和xdiff为4维ND格式，rois为2维ND格式。
- xdiff_shape属性必须为4元素正整数列表。
- 多个采样点映射到同一输入像素时使用原子加累加（非确定性计算）。
- 空tensor（shape含0维）时返回全零梯度。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn 接口调用 | [test_aclnn_roi_align_v2_backward_l2](examples/test_aclnn_roi_align_v2_backward.cpp) | 通过[aclnnRoiAlignV2Backward](./docs/aclnnRoiAlignV2Backward.md)接口方式调用RoiAlignGrad算子。 |
