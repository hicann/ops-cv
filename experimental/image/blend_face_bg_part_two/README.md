# BlendFaceBgPartTwo

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |

## 功能说明

- 算子功能：根据人脸区域累积图、累积 mask、最大 mask 与背景图计算融合结果。
- 计算公式：

  ```text
  fused_img = (acc_face / (acc_mask + epsilon)) * max_mask + bg_img * (1 - max_mask)
  ```

  其中 `epsilon` 为数值稳定项，默认值为 `1e-12`。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 360px">
  <col style="width: 180px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>acc_face</td>
      <td>输入</td>
      <td>人脸区域累积图。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>acc_mask</td>
      <td>输入</td>
      <td>人脸区域累积 mask。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_mask</td>
      <td>输入</td>
      <td>融合权重 mask，取值通常位于 [0, 1]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bg_img</td>
      <td>输入</td>
      <td>背景图。</td>
      <td>FLOAT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>除法数值稳定项，默认值为 1e-12。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fused_img</td>
      <td>输出</td>
      <td>融合后的输出图像。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `acc_face`、`acc_mask`、`max_mask`、`bg_img` 和 `fused_img` 的 shape 需保持一致。
- `acc_face`、`acc_mask`、`max_mask` 和 `fused_img` 目前只支持 FLOAT32。
- `bg_img` 支持 FLOAT32 和 UINT8。
- 当前仅支持 ND 格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn 调用 | examples/test_aclnn_blend_face_bg_part_two.cpp | aclnn 调用示例。 |
