# PasteSubImg

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

PasteSubImg 是一个面向图像或特征图 patch 拼接场景的区域累加算子，计算范式归为 layout_transform。算子从源图像 `patch_img` 中按 `indices` 指定的矩形子区域提取像素，经 `scale` 坐标缩放与 `offsets` 给出的平移量映射到目标画布 `combine_img` 的对应矩形位置，对目标区域内每个像素执行逐元素累加（`combine_img[dst] += patch_img[src]`），目标区域之外的像素保持原值。`combine_img` 以原地方式同时作为输入与输出，适用于图像拼接、超分辨率回填、滑窗推理特征图聚合等需要重叠区域累加的场景。

计算公式如下，其中 `offsets = [px1, py1, px2, py2]`（仅 px1、py1 参与计算），`indices = [cx1, cy1, cx2, cy2]`（右/下边界 exclusive），scale 为坐标缩放因子：

$$
s_{cy1}=\lfloor cy1 \cdot scale \rfloor,\ s_{cy2}=\lfloor cy2 \cdot scale \rfloor,\ s_{cx1}=\lfloor cx1 \cdot scale \rfloor,\ s_{cx2}=\lfloor cx2 \cdot scale \rfloor
$$

$$
d_{cy1}=\lfloor (cy1+py1) \cdot scale \rfloor,\ d_{cx1}=\lfloor (cx1+px1) \cdot scale \rfloor
$$

$$
combine\_img\_out[h, w, c] =
\begin{cases}
combine\_img[h, w, c] + patch\_img[s_{cy1}+(h-d_{cy1}),\ s_{cx1}+(w-d_{cx1}),\ c] & (h,w) \in [d_{cy1},\ d_{cy1}+\Delta h) \times [d_{cx1},\ d_{cx1}+\Delta w) \\
combine\_img[h, w, c] & \text{otherwise}
\end{cases}
$$

其中 $\Delta h = s_{cy2} - s_{cy1}$、$\Delta w = s_{cx2} - s_{cx1}$，C 维完全对齐逐元素相加。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>patch_img</td>
<td>输入</td>
<td>源图像或特征图 patch，3D (H, W, C)，HWC 行优先；对应公式中 patch_img。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>offsets</td>
<td>输入</td>
<td>目标位置偏移 [px1, py1, px2, py2]，1D (4,)；仅 px1（列偏移）与 py1（行偏移）参与计算，对应公式中 offsets。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>indices</td>
<td>输入</td>
<td>源提取区域 [cx1, cy1, cx2, cy2]，1D (4,)，右/下边界 exclusive；对应公式中 indices。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img</td>
<td>输入</td>
<td>目标画布，3D (H_out, W_out, C)，HWC 行优先；以原地方式同时作为输入与输出，累加结果直接写回该张量。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img_out</td>
<td>输出</td>
<td>累加后的画布，与 combine_img 同形同数据类型；aliasing: inplace_with(combine_img)。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>scale</td>
<td>可选属性</td>
<td>坐标缩放因子，乘到所有坐标值上把坐标空间映射到像素空间；默认 1.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- patch_img 与 combine_img 的数据类型必须一致，C 维度必须相同。
- offsets 与 indices 的 shape 均为 [4]；indices 须满足 cx2 > cx1 且 cy2 > cy1。
- coord·scale 不越界 patch_img 维度；目标区域 (coord + offsets)·scale 不越界 combine_img 维度。
- scale 取值范围为 [0.0, 256.0]，非整数结果截断为 int 索引。
- 数据布局为 HWC 行优先 ND，不支持 NCHW 等其他布局。
- combine_img 以原地方式写回，调用前需保证为连续内存。
- 不支持空 Tensor（各维度长度须 ≥ 1）。
- uint8 路径经 float16 中转完成加法，溢出按饱和处理（>255 截断为 255），与部分框架的回绕语义不同。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | - | 通过 [算子IR](op_graph/paste_sub_img_proto.h) 构图调用。 |
