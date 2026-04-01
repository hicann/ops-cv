# RoiPoolingWithArgMax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √   |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对输入特征图按 ROI（感兴趣区域）进行池化，在每个 ROI 内按空间划分为 `pooled_h × pooled_w` 个格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引（argmax）。

- 计算公式：

  输入特征图 $x$ 的 shape 为 $(N, C, H, W)$，ROI 张量 $\text{rois}$ 的 shape 为 $(\text{num\_rois}, 5)$，每行表示 $(b_n, x_1, y_1, x_2, y_2)$。标量参数为 $s_h$、$s_w$（spatial_scale）以及 $\text{pooled\_h}$、$\text{pooled\_w}$。下标 $n$ 表示 ROI 索引，$c$ 表示通道，$(\text{ph}, \text{pw})$ 表示池化格点。

  - **ROI 映射到特征图**：将 ROI 坐标乘以 spatial_scale 得到特征图上的浮点区间：

    $$
    \tilde{x}_1 = x_1 s_w,\quad \tilde{y}_1 = y_1 s_h,\quad \tilde{x}_2 = (x_2+1)s_w,\quad \tilde{y}_2 = (y_2+1)s_h
    $$

    $$
    W_{\text{roi}} = \tilde{x}_2 - \tilde{x}_1,\qquad H_{\text{roi}} = \tilde{y}_2 - \tilde{y}_1
    $$

    若 $W_{\text{roi}} \le 0$ 或 $H_{\text{roi}} \le 0$，该 ROI 的 $y$ 全为 0，$\text{argmax}$ 全为 -1。

  - **Bin 步长与区间**：每个池化格 (ph, pw) 对应 ROI 内一个 bin，步长与浮点区间为：

    $$
    \Delta w = \frac{W_{\text{roi}}}{\text{pooled\_w}},\qquad \Delta h = \frac{H_{\text{roi}}}{\text{pooled\_h}}
    $$

    $$
    \tilde{w}_1 = \text{pw} \cdot \Delta w + \tilde{x}_1,\quad \tilde{w}_2 = (\text{pw}+1) \cdot \Delta w + \tilde{x}_1
    $$

    $$
    \tilde{h}_1 = \text{ph} \cdot \Delta h + \tilde{y}_1,\quad \tilde{h}_2 = (\text{ph}+1) \cdot \Delta h + \tilde{y}_1
    $$

    取整并裁剪到 $[0,W) \times [0,H)$：

    $$
    w_1 = \text{clip}(\lfloor\tilde{w}_1\rfloor,\, 0,\, W),\quad w_2 = \text{clip}(\lceil\tilde{w}_2\rceil,\, 0,\, W)
    $$

    $$
    h_1 = \text{clip}(\lfloor\tilde{h}_1\rfloor,\, 0,\, H),\quad h_2 = \text{clip}(\lceil\tilde{h}_2\rceil,\, 0,\, H)
    $$

    其中 $\text{clip}(a,l,u) = \min(\max(a,l), u)$。若 $w_2 \le w_1$ 或 $h_2 \le h_1$，该 bin 为空：$y=0$，$\text{argmax}=-1$。

  - **池化输出与 Argmax**：记 $b = \text{rois}[n,0]$，bin 区域 $R = \{(h,w) : h_1 \le h < h_2,\, w_1 \le w < w_2\}$，则

    $$
    y[n,c,\text{ph},\text{pw}] = \max_{(h,w) \in R} x[b,c,h,w]
    $$

    （空 $R$ 时为 0。）

    $$
    \text{argmax}[n,c,\text{ph},\text{pw}] = h^* W + w^*
    $$

    $(h^*, w^*)$ 为 bin 内最大值位置（多解取第一个）；空 $R$ 为 -1。

  - **输出 Shape**：

    | 输出 | Shape | 数据类型 |
    |------|--------|----------|
    | $y$ | $(\text{num\_rois},\, C,\, \text{pooled\_h},\, \text{pooled\_w})$ | 与 $x$ 一致 |
    | $\text{argmax}$ | 同上 | INT32 |

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
      <td>x</td>
      <td>输入</td>
      <td>输入特征图。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI边界框。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_actual_num</td>
      <td>可选输入</td>
      <td>指定每个batch的ROI数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化结果。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输出</td>
      <td>每个池化格点最大值在通道内的线性偏移索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* x、rois、y、argmax 的数据类型或格式在支持的范围之内。
* x 的 shape 是 4 维（NCHW）。
* rois 的 shape 第二维是 5。
* pooled_h、pooled_w、spatial_scale_h、spatial_scale_w 大于 0。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_roi_pooling_with_arg_max](examples/arch35/test_aclnn_roi_pooling_with_arg_max.cpp) | 通过[aclnnRoiPoolingWithArgMax](docs/aclnnRoiPoolingWithArgMax.md)接口方式调用RoiPoolingWithArgMax算子。 |
