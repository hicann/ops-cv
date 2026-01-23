# ResizeBilinearV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：使用双线性插值调整图像大小到指定的大小。

- 计算公式：

  假设x中已知四个点$Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$的值，下面对数据进行插值，得到y中的一个点坐标$P(a, b)$中的值。
  
  $$
  f(R_1) = \frac{x_2 - a}{x_2 - x_1}f(Q_{11}) + \frac{a - x_1}{x_2 - x_1}f(Q_{21})
  $$

  $$
  f(R_2) = \frac{x_2 - a}{x_2 - x_1}f(Q_{12}) + \frac{a - x_1}{x_2 - x_1}f(Q_{22})
  $$

  $$
  f(P) = \frac{y_2 - b}{y_2 - y_1}f(R_{1}) + \frac{b - y_1}{y_2 - y_1}f(R_{2})
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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中x。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>单线性插值调整后图像，对应公式中y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_resize](examples/test_aclnn_resize.cpp) | 通过[aclnnResize](docs/aclnnResize.md)接口方式调用ResizeBlinearV2算子。 |