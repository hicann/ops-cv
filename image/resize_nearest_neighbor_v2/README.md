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

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。如果输入shape为（N，C，H, W），则输出shape为（N，C，size[0], size[1]）。
- 计算公式：

  $$
  y(N, C, H, W) = x(N, C, min(floor(H * scaleH),  H-1), min(floor(W * scaleW),  W-1)), \ scaleH = x\_H / size[0] scaleW = x\_W / size[1]
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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中`x`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>最近邻插值后的图像，对应公式中的`y`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody>
  </table>
  
## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                      |
| -------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn  | [test_aclnn_upsample_nearest2d.cpp](examples/test_aclnn_upsample_nearest2d.cpp) | 通过[aclnnUpsampleNearest2d](docs/aclnnUpsampleNearest2d.md)接口方式调用ResizeNearestNeighborV2算子。 |