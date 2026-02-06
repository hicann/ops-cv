# ResizeNearestNeighborV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。如果输入shape为（N，C，L），则输出shape为（N，C，outputSize）。
- 计算公式：

  $$
  out(N, C, l) = self(N, C, min(floor(l * scale),  L-1)), \ scale = self\_L / outputSize[0]
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 223px">
  <col style="width: 101px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示进行上采样的输入数据，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE、UINT8</td>
      <td>NCL</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>表示输出out在L维度上的空间大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示进行上采样的输出结果，对应公式中的`out`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE、UINT8</td>
      <td>NCL</td>
    </tr>
  </tbody>
  </table>
## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                      |
| -------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn  | [test_aclnn_upsample_nearest1d.cpp](examples/test_aclnn_upsample_nearest1d.cpp) | 通过[aclnnUpsampleNearest1d](docs/aclnnUpsampleNearest1d.md)接口方式调用ResizeNearestNeighborV2算子。 |