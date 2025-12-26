# UpsampleTrilinear3dBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：[ResizeUpsampleTrilinear](../resize_upsample_trilinear/README.md)的反向计算。

- 计算公式：
  - 正向核心算法逻辑：
    1. 将目标图像缩放到和原始图像一样大的尺寸。
    2. 计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
    
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时有不同。则有以下公式：
    
    $$
    scale\_d =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[4]-1) / (outputSize[2]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$
   
    那么，对于output的某个方向上的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系：
    
    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$
    
    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$
    
    - 记：
    
      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

      $$
      z_{0} =int(z'),z_{1} =int(z')+1, lambdc_{0} = z_{1}-z', lambdc_{1} =   1-lambdc_{0}
      $$
    
    - 则有以下公式：

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}} 
      $$

    - 假设：正向插值的输出图像out $(x, y, z)$受原图像input $(x_i, y_j, z_k)$影响，则有:
  
      $$
      gradInput(x_i,y_j,z_k) += gradOutput(x,y,z) * lambd(x_i,y_j,z_k)
      $$

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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOutput`。</td>
      <td>FLOAT32、FLOAT16、DOUBLE</td>
      <td>NCDHW</td><!--aclnn多增了一个NCHW-->
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>表示输出`gradInput`分别在N、C、D、H和W维度上的空间大小，对应公式中的`inputSize`。size为5，且各元素均大于零。 </td><!--这个IR原型和aclnn中的解释有冲突，待确认-->
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td><!--aclnn是必选输入-->
      <td>表示输入`grad_output`在D、H和W维度上的空间大小，对应公式中的`outputSize`。size为3，且各元素均大于零。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>指定沿每个维度的缩放数组，包含3个元素：scale_depth, scale_height, scale_width，对应公式中的`scales_d`、`scales_h`、`scales_w`。</li><li>默认值为空。</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td><!--aclnn是必选输入-->
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。数据类型和数据格式与入参`grad_output`的数据类型和数据格式保持一致。shape依赖`input_size`和`output_size/scales`。</td>
      <td>FLOAT32、FLOAT16、DOUBLE</td><!--aclnn多了数据类型bf16-->
      <td>NCDHW</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_trilinear3d_backward](examples/test_aclnn_upsample_trilinear3d_backward.cpp) | 通过[aclnnUpsampleTrilinear3dBackward](docs/aclnnUpsampleTrilinear3dBackward.md)接口方式调用UpsampleTrilinear3dBackward算子。 |
<!--| 图模式 | [test_geir_upsample_trilinear3d_backward](examples/test_geir_upsample_trilinear3d_backward.cpp)  | 通过[算子IR](op_graph/upsample_trilinear3d_backward_proto.h)构图方式调用UpsampleTrilinear3dBackward算子。         |-->