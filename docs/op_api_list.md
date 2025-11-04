# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

算子接口列表如下：

| 接口名                                                       | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [aclnnGridSampler2D](../image/grid_sample/docs/aclnnGridSampler2D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |
| [aclnnGridSampler3D](../image/grid_sample/docs/aclnnGridSampler3D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |
| [aclnnGridSampler2DBackward](../image/grid_sampler2_d_grad/docs/aclnnGridSampler2DBackward.md) | [aclnnGridSampler2D](../image/grid_sample/docs/aclnnGridSampler2D.md)的反向传播，完成张量input与张量grid的梯度计算。 |
| [aclnnGridSampler3DBackward](../image/grid_sampler3_d_grad/docs/aclnnGridSampler3DBackward.md) | [aclnnGridSampler3D](../image/grid_sample/docs/aclnnGridSampler3D.md)的反向传播，完成张量input与张量grid的梯度计算。 |
| [aclnnResize](../image/resize_bilinear_v2/docs/aclnnResize.md) | 根据scales调整输入张量的大小。                               |
| [aclnnUpsampleNearest1d](../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |
| [aclnnUpsampleNearest2d](../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |
| [aclnnUpsampleTrilinear3d](../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md) | 对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。 |
| [aclnnThreeInterpolateBackward](../image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward.md) | 根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。      |
| [aclnnUpsampleBicubic2d](../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md) | 对由多个输入通道组成的输入信号应用2D双三次上采样。           |
| [aclnnUpsampleBicubic2dAA](../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md) | 对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。 |
| [aclnnUpsampleBicubic2dAAGrad](../image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md) | [aclnnUpsampleBicubic2dAA](../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。 |
| [aclnnUpsampleBicubic2dBackward](../image/upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md) | [aclnnUpsampleBicubic2d](../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)的反向传播。 |
| [aclnnUpsampleBilinear2d](../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md) | 对由多个输入通道组成的输入信号应用2D双线性上采样。           |
| [aclnnUpsampleBilinear2dAA](../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md) | 对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。       |
| [aclnnUpsampleBilinear2dAABackward](../image/upsample_bilinear2d_aa_backward/docs/aclnnUpsampleBilinear2dAABackward.md) | [aclnnUpsampleBilinear2dAA](../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md)的反向传播。 |
| [aclnnUpsampleBilinear2dBackwardV2](../image/upsample_bilinear2d_grad/docs/aclnnUpsampleBilinear2dBackwardV2.md) | [aclnnUpsampleBilinear2d](../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。 |
| [aclnnUpsampleLinear1d](../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md) | 对由多个输入通道组成的输入信号应用线性插值算法进行上采样。   |
| [aclnnUpsampleLinear1dBackward](../image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md) | [aclnnUpsampleLinear1d](../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md)的反向传播。 |
| [aclnnUpsampleNearestExact1d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md) | 对由三个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。 |
| [aclnnUpsampleNearestExact2d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md) | 对由四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。 |
| [aclnnUpsampleNearest1dBackward](../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward.md) | [aclnnUpsampleNearestExact1d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md)的反向传播。 |
| [aclnnUpsampleNearest2dBackward](../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward.md) | [aclnnUpsampleNearestExact2d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md)的反向传播。 |
| [aclnnUpsampleNearest1dV2](../image/upsample_nearest3d/docs/aclnnUpsampleNearest1dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |
| [aclnnUpsampleNearest2dV2](../image/upsample_nearest3d/docs/aclnnUpsampleNearest2dV2.md) | 对由四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。 |
| [aclnnUpsampleNearest3d](../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |
| [aclnnUpsampleNearest3dBackward](../image/upsample_nearest3d_grad/docs/aclnnUpsampleNearest3dBackward.md) | [aclnnUpsampleNearest3d](../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md)的反向传播。 |
| [aclnnUpsampleNearestExact1dBackward](../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact1dBackward.md) | [aclnnUpsampleNearestExact1d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md)的反向传播。 |
| [aclnnUpsampleNearestExact2dBackward](../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact2dBackward.md) | [aclnnUpsampleNearestExact2d](../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md)的反向传播。 |
| [aclnnUpsampleNearestExact3d](../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |
| [aclnnUpsampleNearestExact3dBackward](../image/upsample_nearest_exact3d_grad/docs/aclnnUpsampleNearestExact3dBackward.md) | [aclnnUpsampleNearestExact3d](../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md)的反向传播。 |
| [aclnnUpsampleTrilinear3dBackward](../image/upsample_trilinear3d_backward/docs/aclnnUpsampleTrilinear3dBackward.md) | [aclnnUpsampleTrilinear3d](../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md)的反向传播。 |
| [aclnnIou](../objdetect/iou_v2/docs/aclnnIou.md)          | 计算两组矩形框（预测框bBox与真值框gtBox）的交并比（IOU）或前景交叉比（IOF），用于评估其重叠程度。 |
| [aclnnRoiAlign](../objdetect/roi_align/docs/aclnnRoiAlign.md) | RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |
| [aclnnRoiAlignV2](../objdetect/roi_align/docs/aclnnRoiAlignV2.md) | RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |
| [aclnnRoiAlignV2Backward](../objdetect/roi_align_grad/docs/aclnnRoiAlignV2Backward.md) | [aclnnRoiAlignV2](../objdetect/roi_align/docs/aclnnRoiAlignV2.md)的反向传播。 |