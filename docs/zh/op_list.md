# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>image</td>
    <td><a href="../../image/crop_and_resize/README.md">crop_and_resize</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>从输入图像中提取多个裁剪区域,并将它们统一调整为指定大小，支持双线性插值和最近邻插值。  </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/deformable_offsets/README.md">deformable_offsets</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>用于计算变形卷积（Deformable Convolution）输出的函数。通过引入偏移参数 offsets ，使得卷积核在输入特征图上的位置可以动态调整，从而适配不规则的集合变化。  </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/deformable_offsets_grad/README.md">deformable_offsets_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>DeformableOffsets 算子的目的是根据 offsets（ kernel 采样点的偏移值）来收集用于卷积的特征采样点，并对其进行重组，方便 Conv2d 算子进行卷积计算。而 DeformableOffsetsGrad 即为这一过程的反向。  </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sample/README.md">grid_sample</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler2_d_grad/README.md">grid_sampler2_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>GridSampler中2D场景的反向传播，完成张量input与张量grid的梯度计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler3_d_grad/README.md">grid_sampler3_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>GridSampler中3D场景的反向传播，完成张量input与张量grid的梯度计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/image_warp_offsets/README.md">image_warp_offsets</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>根据偏移量选取图像并进行扭曲变换。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/non_max_suppression_v3/README.md">non_max_suppression_v3</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>按照分数递减顺序，采用贪心策略选择候选框（bounding boxes）子集。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2/README.md">resize_bicubic_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>使用双三次插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2_grad/README.md">resize_bicubic_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>计算输入图像在双三次插值基础下的梯度。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2/README.md">resize_bilinear_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>使用双线性插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2_grad/README.md">resize_bilinear_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>ResizeBilinearV2的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear/README.md">resize_linear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>使用单线性插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear_grad/README.md">resize_linear_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>计算输入图像在单线性插值基础下的梯度。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_nearest_neighbor_v2/README.md">resize_nearest_neighbor_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
    <tr>
    <td>image</td>
    <td><a href="../../image/resize_upsample_trilinear/README.md">resize_upsample_trilinear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/three_interpolate_backward/README.md">three_interpolate_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据grad_x,idx,weight进行三点插值计算梯度得到grad_y。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d/README.md">upsample_bicubic2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双三次上采样。如果输入Tensorx的shape为(N,C,H,W)，则输出Tensorout的shape为(N,C,outputSize[0],outputSize[1])。 </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa/README.md">upsample_bicubic2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入Tensorx的shape为(N,C,H,W)，则输出Tensorout的shape为(N,C,outputSize[0],outputSize[1])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa_grad/README.md">upsample_bicubic2d_aa_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>如果输入张量grad_output的shape为(N,C,H,W)，则输出张量grad_input的shape为(N,C,inputSize[2],inputSize[3])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_grad/README.md">upsample_bicubic2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>如果输入张量grad_output的shape为(N,C,H,W)，则输出张量grad_input的shape为(N,C,inputSize[2],inputSize[3])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d/README.md">upsample_bilinear2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双线性上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa/README.md">upsample_bilinear2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa_backward/README.md">upsample_bilinear2d_aa_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleBilinear2dAA的反向传播。 </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_grad/README.md">upsample_bilinear2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleBilinear2d的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_linear1d/README.md">upsample_linear1d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用线性插值算法进行上采样。如果输入shape为（N，C，L），则输出shape为（N，C，outputSize）。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest/README.md">upsample_nearest</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact2d_grad/README.md">upsample_nearest_exact2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>[UpsampleNearest](../../image/upsample_nearest/README.md)在exact_mode为true时的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d/README.md">upsample_nearest_exact3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d_grad/README.md">upsample_nearest_exact3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleNearestExact3d的反向计算。  </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest2d_grad/README.md">upsample_nearest2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleNearest在exact_mode为false时的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d/README.md">upsample_nearest3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d_grad/README.md">upsample_nearest3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleNearest3d的反向计算。 </td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_trilinear3d_backward/README.md">upsample_trilinear3d_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>ResizeUpsampleTrilinear的反向计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/mrgba_custom/doc/aclnnMrgbaCustom.md">mrgba_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb和张量alpha的透明度乘法计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/blend_images_custom/doc/aclnnBlendImagesCustom.md">blend_images_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb、frame和alpha的透明度乘法计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/background_replace/doc/aclnnBackgroundReplace.md">background_replace</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/iou_v2/README.md">iou_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算两个矩阵的重叠面积占两个矩阵总面积的比例，设预测框的左上角坐标为（X1，Y1），右下角坐标为（X2，Y2），真实框的左上角坐标为（X3，Y3），右下角坐标为（X4，Y4）。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/non_max_suppression_v6/README.md">non_max_suppression_v6</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align/README.md">roi_align</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_grad/README.md">roi_align_grad</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated/README.md">roi_align_rotated</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>用于旋转候选框的ROI对齐池化层。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated_grad/README.md">roi_align_rotated_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>通过旋转框各点坐标将梯度回传至对应位置。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/stack_group_points/README.md">stack_group_points</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据特征点所属的组，重组点云中的特征点。 </td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/background_replace/README.md">background_repalce</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/blend_images_custom/README.md">blend_images_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb、frame和alpha的透明度乘法计算。 </td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/mrgba_custom/README.md">mrgba_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb和张量alpha的透明度乘法计算。 </td>
  </tr>
</tbody>
</table>
