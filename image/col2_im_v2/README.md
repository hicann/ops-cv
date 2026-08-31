# Col2ImV2

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

- 算子功能：将滑窗局部块（column）重排组合为批图像张量，即col2im/fold计算。输出采用输出centric方式滑窗，多个滑窗覆盖同一输出位置时累加，未被任何滑窗覆盖的输出位置为0。

- 计算公式：

$$
ho = \left\lfloor \frac{outH + 2 \times padH - dilH \times (kH - 1) - 1}{strideH} \right\rfloor + 1
$$

$$
y[n,c,h,w] = \sum_{h\_col,w\_col,h\_k,w\_k} x[n,\ (c \times kH + h\_k) \times kW + w\_k,\ h\_col \times wo + w\_col]
$$

其中求和下标需满足$h = h\_col \times strideH - padH + h\_k \times dilH$，$w$同理；$wo$的计算公式与$ho$对称。

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
      <td>x</td>
      <td>输入</td>
      <td>待进行col2im计算的滑窗局部块，公式中的x，shape为(N, C×kH×kW, L)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>输入</td>
      <td>输出图像尺寸(outH, outW)，长度为2的const tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernel_size</td>
      <td>输入</td>
      <td>滑窗尺寸(kH, kW)，长度为2的const tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>属性</td>
      <td>滑窗空洞间隔(dilH, dilW)，长度为2的ListInt，元素取值大于0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>边缘填充大小(padH, padW)，长度为2的ListInt，元素取值大于等于0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>属性</td>
      <td>滑窗步长(strideH, strideW)，长度为2的ListInt，元素取值大于0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>col2im计算结果，公式中的y，shape为(N, C, outH, outW)，dtype与x相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x仅支持3维tensor，shape为(N, C×kH×kW, L)；y仅支持4维tensor，shape为(N, C, outH, outW)。
- x与y同dtype，仅支持float32和float16。
- output_size和kernel_size仅支持1维const tensor，长度必须为2，dtype仅支持int32；其取值在编译期必须已知（值依赖输入）。
- output_size的元素值必须大于0；kernel_size的元素值必须大于0。
- dilation长度为2且元素大于0；padding长度为2且元素大于等于0；stride长度为2且元素大于0。
- 跨参数约束：x的第1维必须能被kH×kW整除；x的第2维L必须等于ho×wo，其中ho=(outH+2×padH-dilH×(kH-1)-1)//strideH+1，wo=(outW+2×padW-dilW×(kW-1)-1)//strideW+1。
- 未被任何滑窗覆盖的输出位置取值为0；多个滑窗覆盖同一输出位置时取值累加。

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
    <td><a href="./examples/test_geir_col2_im_v2.cpp">test_geir_col2_im_v2</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
