# ResizeGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：计算Resize正向算子的反向梯度，将resize后的梯度grads按linear（1D）或cubic（2D）插值权重累加回原始分辨率的每个采样位置，输出与正向输入同shape的梯度y。

- 计算公式（cubic模式，输出点$[h,w]$反向聚合覆盖它的输入窗口）：

$$
y[n,c,h,w] = \sum_{(o_h,o_w)\in S(h,w)} grads[n,c,o_h,o_w] \cdot c_h(o_h) \cdot c_w(o_w)
$$

  其中$c_h$、$c_w$为三次卷积插值系数（系数a固定为-0.75），$S(h,w)$为覆盖输出点$[h,w]$的输入采样窗口；linear模式为1D线性插值的两点加权累加；输入输出尺寸相同时退化为纯拷贝。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 190px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 230px">
  <col style="width: 150px">
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
      <td>grads</td>
      <td>输入</td>
      <td>resize后的梯度，4维Tensor。cubic模式NCHW布局为(N,C,H_out,W_out)，HWNC布局为(H_out,W_out,N,C)；linear模式为(N,C,1,W_out)。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi</td>
      <td>输入（可选）</td>
      <td>1维Tensor，仅在coordinate_transformation_mode为"tf_crop_and_resize"时生效，本次不支持该模式，不使用。</td>
      <td>FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入（可选）</td>
      <td>1维float Tensor，各维度缩放系数，cubic模式2元、linear模式1元。元素大于0时生效（scale=1/scales[i]），缺省或元素不大于0时按尺寸推导。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_size</td>
      <td>输入</td>
      <td>1维int32/int64 Tensor，输出Tensor的尺寸。cubic模式4元[N,C,H_in,W_in]（NCHW语义），linear模式3元[N,C,W_in]。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>coordinate_transformation_mode</td>
      <td>属性</td>
      <td>resize后坐标到原始坐标的变换方式，本次仅支持"half_pixel"和"align_corners"，默认"half_pixel"。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubic_coeff_a</td>
      <td>属性</td>
      <td>cubic插值系数，仅cubic模式使用，本次固定-0.75。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclude_outside</td>
      <td>属性</td>
      <td>保留属性，默认0，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>extrapolation_value</td>
      <td>属性</td>
      <td>保留属性，默认0.0，仅"tf_crop_and_resize"模式生效，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>插值模式，本次仅支持"linear"和"cubic"，默认"nearest"但不支持（传入即报错，对齐ResizeGradD的NPU行为）。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nearest_mode</td>
      <td>属性</td>
      <td>保留属性，默认"round_prefer_floor"，仅nearest模式使用，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>计算时输入数据的布局，支持"NCHW"和"HWNC"，默认"NCHW"；linear模式仅支持"NCHW"。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向输入的梯度，shape由original_size推导，dtype与grads相同。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- grads与y仅支持FLOAT16、FLOAT32两种dtype，且y.dtype与grads.dtype一致。
- mode仅支持"linear"和"cubic"，"nearest"及其他取值不支持（对齐ResizeGradD的NPU行为）。
- data_format支持"NCHW"和"HWNC"，linear模式仅支持"NCHW"。
- grads必须为4维：cubic模式NCHW布局为(N,C,H_out,W_out)，HWNC布局为(H_out,W_out,N,C)；linear模式为(N,C,1,W_out)且第3维（dim(2)）必须为1。
- original_size为值依赖输入：各元素必须大于0，前2维（N、C）须与grads一致；cubic模式必须为4元[N,C,H_in,W_in]，linear模式必须为3元[N,C,W_in]。
- scales可选：cubic模式2元、linear模式1元；元素大于0时生效（scale=1/scales[i]），缺省或元素不大于0时按输入输出尺寸推导；linear模式下scales[0]须约等于grads.W/original_size.W（容差1e-4）。
- coordinate_transformation_mode仅支持"half_pixel"和"align_corners"；cubic_coeff_a固定-0.75。
- roi、exclude_outside、extrapolation_value、nearest_mode为保留参数，本次不参与计算。
- 本实现为确定性实现（gather固定顺序累加），与scatter语义数学等价。
- 测试说明：白盒idx64用例03_resize_grad_whitebox_idx64_tse.csv（378条长耗时用例）未纳入TTK主流程，按专用长耗时档位单独调度。

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
    <td><a href="./examples/test_geir_resize_grad.cpp">test_geir_resize_grad</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
