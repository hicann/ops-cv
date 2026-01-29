# 算子接口（aclnn）

##  使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓（如math、nn、cv、transformer），请配置为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

| 接口名      | 说明     | 确定性说明（A2/A3） | 确定性说明（A5） |
| -------------- | --------------------------- | --------------------------- | --------------------------- |
| [aclnnMrgbaCustom](../../objdetect/mrgba_custom/doc/aclnnMrgbaCustom.md) | 完成张量rgb和张量alpha的透明度乘法计算。 |默认确定性实现|
| [aclnnBackgroundReplace](../../objdetect/background_replace/doc/aclnnBackgroundReplace.md) | 将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。 |默认确定性实现|
| [aclnnBlendImagesCustom](../../objdetect/blend_images_custom/doc/aclnnBlendImagesCustom.md) | 完成张量rgb、frame和alpha的透明度乘法计算。 |默认确定性实现|
| [aclnnGridSampler2D](../../image/grid_sample/docs/aclnnGridSampler2D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|
| [aclnnGridSampler3D](../../image/grid_sample/docs/aclnnGridSampler3D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|
| [aclnnGridSampler2DBackward](../../image/grid_sampler2_d_grad/docs/aclnnGridSampler2DBackward.md) | [aclnnGridSampler2D](../../image/grid_sample/docs/aclnnGridSampler2D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|
| [aclnnGridSampler3DBackward](../../image/grid_sampler3_d_grad/docs/aclnnGridSampler3DBackward.md) | [aclnnGridSampler3D](../../image/grid_sample/docs/aclnnGridSampler3D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|
| [aclnnResize](../../image/resize_bilinear_v2/docs/aclnnResize.md) | 根据scales调整输入张量的大小。                               |默认确定性实现|
| [aclnnThreeInterpolateBackward](../../image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward.md) | 根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。      |默认非确定性实现，不支持配置开启|
| [aclnnUpsampleNearest1d](../../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearest2d](../../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleTrilinear3d](../../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md) | 对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleBicubic2d](../../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md) | 对由多个输入通道组成的输入信号应用2D双三次上采样。           |默认确定性实现|
| [aclnnUpsampleBicubic2dAA](../../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md) | 对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleBicubic2dAAGrad](../../image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md) | [aclnnUpsampleBicubic2dAA](../../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleBicubic2dBackward](../../image/upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md) | [aclnnUpsampleBicubic2d](../../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)的反向传播。 |默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md) | 对由多个输入通道组成的输入信号应用2D双线性上采样。|默认确定性实现|
| [aclnnUpsampleBilinear2dAA](../../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md) | 对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。|默认确定性实现|
| [aclnnUpsampleBilinear2dAABackward](../../image/upsample_bilinear2d_aa_backward/docs/aclnnUpsampleBilinear2dAABackward.md) | [aclnnUpsampleBilinear2dAA](../../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleBilinear2dBackward](../../image/resize_bilinear_v2_grad/docs/aclnnUpsampleBilinear2dBackward.md) | [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleBilinear2dBackwardV2](../../image/upsample_bilinear2d_grad/docs/aclnnUpsampleBilinear2dBackwardV2.md) | [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleLinear1d](../../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md) | 对由多个输入通道组成的输入信号应用线性插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleLinear1dBackward](../../image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md) | [aclnnUpsampleLinear1d](../../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleNearestExact1d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearestExact2d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearest1dBackward](../../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward.md) | [aclnnUpsampleNearestExact1d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleNearest2dBackward](../../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward.md) | [aclnnUpsampleNearestExact2d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md)的反向传播。 |默认确定性实现|
| [aclnnUpsampleNearest1dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest1dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearest2dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest2dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearest3d](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearest3dBackward](../../image/upsample_nearest3d_grad/docs/aclnnUpsampleNearest3dBackward.md) | [aclnnUpsampleNearest3d](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md)的反向传播。 |默认非确定性实现，支持配置开启|
| [aclnnUpsampleNearestExact1dBackward](../../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact1dBackward.md) | [aclnnUpsampleNearestExact1d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md)的反向传播。 |默认非确定性实现，支持配置开启|
| [aclnnUpsampleNearestExact2dBackward](../../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact2dBackward.md) | [aclnnUpsampleNearestExact2d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md)的反向传播。 |默认非确定性实现，支持配置开启|
| [aclnnUpsampleNearestExact3d](../../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|
| [aclnnUpsampleNearestExact3dBackward](../../image/upsample_nearest_exact3d_grad/docs/aclnnUpsampleNearestExact3dBackward.md) | [aclnnUpsampleNearestExact3d](../../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md)的反向传播。 |默认非确定性实现，支持配置开启|
| [aclnnUpsampleTrilinear3dBackward](../../image/upsample_trilinear3d_backward/docs/aclnnUpsampleTrilinear3dBackward.md) | [aclnnUpsampleTrilinear3d](../../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md)的反向传播。 |默认确定性实现|
| [aclnnIou](../../objdetect/iou_v2/docs/aclnnIou.md)          | 计算两组矩形框（预测框bBox与真值框gtBox）的交并比（IOU）或前景交叉比（IOF），用于评估其重叠程度。 |默认确定性实现|
| [aclnnNonMaxSuppression](../../objdetect/non_max_suppression_v6/docs/aclnnNonMaxSuppression.md)          | 删除分数小于scoreThreshold的边界框，筛选出与之前被选中部分重叠较高（IOU较高）的框。 |默认确定性实现|
| [aclnnRoiAlign](../../objdetect/roi_align/docs/aclnnRoiAlign.md) | RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|
| [aclnnRoiAlignV2](../../objdetect/roi_align/docs/aclnnRoiAlignV2.md) | RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|
| [aclnnRoiAlignV2Backward](../../objdetect/roi_align_grad/docs/aclnnRoiAlignV2Backward.md) | [aclnnRoiAlignV2](../../objdetect/roi_align/docs/aclnnRoiAlignV2.md)的反向传播。 |默认非确定性实现，支持配置开启|
