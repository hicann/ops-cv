# Col2im

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
## 功能说明

- 算子功能：从批处理输入张量中提取滑动局部块，将滑动局部块数组合并为一个大张量。
- 计算公式：
  
  考虑一个形状为 $(N,C,∗)$的批处理input张量，其中$N$是批处理维度，$C$是通道维度，而$∗$表示任意空间维度。

  此操作将input空间维度内的每个滑动kernel_size大小的块展平为形状是$(N,C×\prod(kernel_size),L)$ 的 3-D output张量的列（即最后一维）。

  其中：
  - $C×\prod(kernel_size)$ 是每个块内的值的数量（一个块有$\prod(kernel_size)$ 个空间位置，每个空间位置都包含一个$C$ 通道向量），而$L$是这些块的总数：

    $$
    L=\prod_d⌊{\frac{spatial\_size[d]+2×padding[d]−dilation[d]×(kernel\_size[d]−1)−1}{stride[d]}+1}⌋
    $$

  - spatial_size由input(上面的$∗$)的空间维度构成，而$d$覆盖所有空间维度。
  因此，在最后一个维度（列维度）索引，output会给出某个块内的所有值。

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
      <td>gradOutput</td>
      <td>输入</td>
      <td>反向输入，对应公式中的output。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>5HD</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>属性</td>
      <td>输入张量的形状，对应公式中的spatial_size。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>属性</td>
      <td>卷积核的大小，对应公式中的kernel_size。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>属性</td>
      <td>膨胀参数，对应公式中的dilation。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>卷积的填充大小，对应公式中的padding。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>属性</td>
      <td>卷积的步长，对应公式中的stride。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>梯度计算结果，对应公式中的input。数据类型和数据格式需要与`gradOutput`的数据类型和数据格式一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>5HD</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_im2col_backward](examples/test_aclnn_im2col_backward.cpp) | 通过[aclnnIm2colBackward](docs/aclnnIm2colBackward.md)接口方式调用Col2im算子。 |
