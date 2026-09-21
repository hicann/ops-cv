# ROIAlign

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：ROI Align(Region of Interest Align)从特征图中提取感兴趣区域(ROI)的特征，通过双线性插值和池化生成固定大小的输出特征图。

- 计算公式：

$$
\text{output}[n, c, ph, pw] = \text{pool}\left(\text{bilinear\_interpolate}(\text{features}[\text{batch\_idx}, c], y, x)\right)
$$

其中坐标映射由`roi_end_mode`属性控制，池化方式由`pool_mode`属性控制（avg/max）。

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
      <td>输入特征图，shape为(N, C, H, W)。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI坐标，shape为(K, 5)，每行[batch_idx, x1, y1, x2, y2]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_n</td>
      <td>输入（可选）</td>
      <td>每个ROI对应的batch数量，shape为(K,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化后的ROI特征，shape为(K, C, pooled_height, pooled_width)。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性（必选）</td>
      <td>坐标缩放因子，将ROI坐标映射到特征图尺度。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>属性（必选）</td>
      <td>输出高度（bin数量）。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>属性（必选）</td>
      <td>输出宽度（bin数量）。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>属性（可选）</td>
      <td>每个bin的采样点数，0表示自适应。默认值2。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>属性（可选）</td>
      <td>坐标变换模式：0=无偏移；1=加1后缩放；>=2=减0.5偏移。默认值1。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_mode</td>
      <td>属性（可选）</td>
      <td>池化模式："avg"（平均池化）或"max"（最大池化）。默认值"avg"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- features 必须为 4D 张量 (N, C, H, W)。
- rois 必须为 2D 张量 (K, 5)，每行格式为 [batch_idx, x1, y1, x2, y2]。
- 0 <= batch_idx <= N - 1, 0 <= x1 < x2, 0 <= y1 < y2。
- features 和 rois 必须使用相同 dtype（float16 或 float32）。
- pooled_height 和 pooled_width 必须 >= 1。
- spatial_scale 必须 > 0。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/arch35/test_geir_roi_align.cpp">test_geir_roi_align</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
