# StackGroupPoints

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 | √ |

## 功能说明

- 算子功能：根据特征点所属的组，重组点云中的特征点。
- 计算公式：点云数据`features`形状为$(N1 + N2 ..., C)$，根据`features_batch_cnt`形状$(N1, N2, ...)$将其分组，同样地，`indices`形状为$(M1 + M2 ..., nsample)$，根据`indices_batch_cnt`形状$(M1, M2, ...)$将其分组。算子根据`indices`分组从对应的`features`组中抽取特征成输出`out`，实现如下功能：
  $$
  out[m:c:np] = features[indices[m:np]:c]
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>公式中的输入features。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>公式中的输入indices。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

待上线

