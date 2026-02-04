# BackgroundReplace

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT   |     ×    |
|  Atlas A3 训练系列产品/Atlas A3 推理系列产品   |     ×    |
|  Atlas A2 训练系列产品/Atlas A2 推理系列产品     |     ×    |
|  Atlas 200I/500 A2 推理产品    |     ×    |
|  Atlas 推理系列产品    |     √   |
|  Atlas 训练系列产品    |     ×    |

## 功能说明

- 算子功能：将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。

- 计算公式：

  $$
  out = bkg * (1 - mask) + src * mask
  $$

  其中，bkg为背景图片，src为源图片，mask为掩码。

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
      <td>bkg</td>
      <td>输入</td>
      <td>背景图片，Device侧的aclTensor，shape支持HWC(C=1、3)。只支持连续Tensor。</td>
      <td>UINT8、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>输入</td>
      <td>源图片，Device侧的aclTensor，shape支持HWC(C=1、3)。只支持连续Tensor。</td>
      <td>UINT8、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>掩码，Device侧的aclTensor，shape支持HWC(C=1)。只支持连续Tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出图片，Device侧的aclTensor，shape支持HWC(C=1、3)，数据类型和shape与输入背景图片bkg一致。只支持连续Tensor。</td>
      <td>UINT8、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* 确定性计算：aclnnBackgroundReplace默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_background_replace](example/test_aclnn_background_replace.cpp) | 通过[aclnnBackgroundReplace](docs/aclnnBackgroundReplace.md)接口方式调用BackgroundReplace算子。 |
