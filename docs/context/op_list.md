# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](dir_structure.md)。
> - **算子执行位置**：大部分算子运行在AI Core上，少部分算子运行在AI CPU上。默认情况下，项目中提到的算子一般指AI Core算子。
> - 关于AI Core和AI CPU详细介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

项目提供的所有算子分类和算子列表如下：

|  算子分类  |   算子目录   |    算子执行位置   | 说明                                                                                                                        |
| --------- | ------------------------------------------------------------ | ---------------- |---------------------------------------------------------------------------------------------------------------------------|
| image | [grid_sample](../../image/grid_sample/README.md) | AI Core          | 提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。                                         |
| image | [grid_sampler2_d_grad](../../image/grid_sampler2_d_grad/README.md) | AI Core          | GridSampler中2D场景的反向传播，完成张量input与张量grid的梯度计算。                                                                              |
| image | [grid_sampler3_d_grad](../../image/grid_sampler3_d_grad/README.md) | AI Core          | GridSampler中3D场景的反向传播，完成张量input与张量grid的梯度计算。                                                                              |
| objdetect | [iou_v2](../../objdetect/iou_v2/README.md) | AI Core          | 计算两个矩阵的重叠面积占两个矩阵总面积的比例，设预测框的左上角坐标为（X1，Y1），右下角坐标为（X2，Y2），真实框的左上角坐标为（X3，Y3），右下角坐标为（X4，Y4）。                                  |
| image | [resize_upsample_trilinear](../../image/resize_upsample_trilinear/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。                                                                                            |
| objdetect | [roi_align_rotated](../../objdetect/roi_align_rotated/README.md) | AI Core          | 用于旋转候选框的ROI对齐池化层。                                                                                                         |
| objdetect | [roi_align_rotated_grad](../../objdetect/roi_align_rotated_grad/README.md) | AI Core          | 通过旋转框各点坐标将梯度回传至对应位置。                                                                                                      |
| objdetect | [stack_group_points](../../objdetect/stack_group_points/README.md) | AI Core          | 根据特征点所属的组，重组点云中的特征点。                                                                                                      |
| image | [three_interpolate_backward](../../image/three_interpolate_backward/README.md) | AI Core          | 根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。                                                                                  |
| image | [upsample_bicubic2d](../../image/upsample_bicubic2d/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用2D双三次上采样。如果输入Tensor x的shape为(N, C, H, W)，则输出Tensor out的shape为(N, C, outputSize[0], outputSize[1])。       |
| image | [upsample_bicubic2d_aa](../../image/upsample_bicubic2d_aa/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入Tensor x的shape为(N, C, H, W) ，则输出Tensor out的shape为(N, C, outputSize[0], outputSize[1])。 |
| image | [upsample_bicubic2d_aa_grad](../../image/upsample_bicubic2d_aa_grad/README.md) | AI Core          | 如果输入张量grad_output的shape为(N, C, H, W)，则输出张量grad_input的shape为(N, C, inputSize[2], inputSize[3])。                            |
| image | [upsample_bicubic2d_grad](../../image/upsample_bicubic2d_grad/README.md) | AI Core          | 如果输入张量grad_output的shape为(N, C, H, W)，则输出张量grad_input的shape为(N, C, inputSize[2], inputSize[3])。                            |
| image | [upsample_bilinear2d](../../image/upsample_bilinear2d/README.md) | AI Core          | 3D场景的反向传播，完成张量input与张量grid的梯度计算。                                                                                          |
| image | [upsample_bilinear2d_aa](../../image/upsample_bilinear2d_aa/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。                                                                                              |
| image | [upsample_bilinear2d_aa_backward](../../image/upsample_bilinear2d_aa_backward/README.md) | AI Core          | UpsampleBilinear2dAA的反向传播。                                                                                                |
| image | [upsample_bilinear2d_grad](../../image/upsample_bilinear2d_grad/README.md) | AI Core          | GridSampler中3D场景的反向传播，完成张量input与张量grid的梯度计算。                                                                              |
| image | [upsample_linear1d](../../image/upsample_linear1d/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用线性插值算法进行上采样。如果输入shape为（N，C，L），则输出shape为（N，C，outputSize）。                                                 |
| image | [upsample_nearest](../../image/upsample_nearest/README.md) | AI Core          | 对由三个或者四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。                                                                                      |
| image | [upsample_nearest2d_grad](../../image/upsample_nearest2d_grad/README.md) | AI Core          | UpsampleNearest在exact_mode为false时的反向传播。                                                                                   |
| image | [upsample_nearest3d](../../image/upsample_nearest3d/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。                                                                                            |
| image | [upsample_nearest3d_grad](../../image/upsample_nearest3d_grad/README.md) | AI Core          | [UpsampleNearest3d](../../image/upsample_nearest3d/README.md)的反向计算。                                            |
| image | [upsample_nearest_exact2d_grad](../../image/upsample_nearest_exact2d_grad/README.md) | AI Core          | [UpsampleNearest](../../image/upsample_nearest/README.md)在exact_mode为true时的反向传播。                                           |
| image | [upsample_nearest_exact3d](../../image/upsample_nearest_exact3d/README.md) | AI Core          | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。                                                                                            |
| image | [upsample_nearest_exact3d_grad](../../image/upsample_nearest_exact3d_grad/README.md) | AI Core          | [UpsampleNearestExact3d](../../image/upsample_nearest_exact3d/README.md)的反向计算。                                      |
| image | [upsample_trilinear3d_backward](../../image/upsample_trilinear3d_backward/README.md) | AI Core          | [ResizeUpsampleTrilinear](../../image/resize_upsample_trilinear/README.md)的反向计算。                                   |
| image     | [crop_and_resize](../../image/crop_and_resize/README.md)     | AI CPU           | 从输入图像中提取多个裁剪区域, 并将他们统一调整为指定大小，支持双线性插值和最近邻插值。                                                                              |
| image     | [image_warp_offsets](../../image/image_warp_offsets/README.md) | AI CPU           | 根据偏移量选取图像并进行扭曲变换。                                                                                                         |