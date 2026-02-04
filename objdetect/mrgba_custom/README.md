# MrgbaCustom

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT   |     是    |
|  Atlas A3 训练系列产品/Atlas A3 推理系列产品   |     是    |
|  Atlas A2 训练系列产品/Atlas A2 推理系列产品     |     是    |
|  Atlas 200I/500 A2 推理产品    |     是    |
|  Atlas 推理系列产品    |     否  |
|  Atlas 训练系列产品    |     是    |

## 功能说明

- 算子功能：完成张量rgb和张量alpha的透明度乘法计算。
- 计算公式：

  $$
  out = rgb * \frac{alpha}{255}
  $$

  其中，alpha会广播到与rgb相同的shape。

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
      <td>rgb</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持HWC(C=3)，与alpha满足broadcast关系，只支持连续Tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持HWC(C=1)，与rgb满足broadcast关系，只支持连续Tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>Device侧的aclTensor，shape支持HWC(C=3)，与rgb的shape一致，只支持连续Tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* 输入shape限制：rgb的shape为HWC(C=3)，alpha的shape为HWC(C=1)，out的shape与rgb一致，且与alpha满足broadcast关系。
* 确定性计算：aclnnMrgbaCustom默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_mrgba_custom](example/test_aclnn_mrgba_custom.cpp) | 通过[aclnnMrgbaCustom](docs/aclnnMrgbaCustom.md)接口方式调用MrgbaCustom算子。 |
