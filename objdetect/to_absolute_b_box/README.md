# ToAbsoluteBBox

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |


## 功能说明

- 算子功能：将归一化边界框坐标（相对坐标，值域[0, 1]）按图像高宽转换为绝对像素坐标。常用于Faster R-CNN、SSD、YOLO等目标检测模型推理后处理阶段，将模型预测的归一化bbox恢复到图像绝对坐标空间。

  bbox的4个坐标分量顺序为`[ymin, xmin, ymax, xmax]`：ymin/ymax乘以图像高度H，xmin/xmax乘以图像宽度W。H取自`shape_hw[1]`、W取自`shape_hw[2]`（`shape_hw`布局为`[batch, H, W, C]`）。属性`reversed_box`控制bbox数据布局，进而决定坐标维（长度为4的轴）所在位置：

  - `reversed_box = false`（默认）：布局`(batch, num, 4)`，坐标维为末轴，缩放系数按坐标顺序为`[H, W, H, W]`。
  - `reversed_box = true`：布局`(batch, 4, num)`，坐标维为次轴，缩放系数按坐标顺序为`[H, W, H, W]`。

- 计算公式：

  $$
  y_{min}^{abs} = y_{min}^{norm} \times H,\quad x_{min}^{abs} = x_{min}^{norm} \times W
  $$

  $$
  y_{max}^{abs} = y_{max}^{norm} \times H,\quad x_{max}^{abs} = x_{max}^{norm} \times W
  $$

  其中H为图像高度、W为图像宽度。

## 参数说明

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 540px">
  <col style="width: 200px">
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
      <td>normalized_boxes</td>
      <td>输入</td>
      <td>归一化边界框坐标，对应公式中的归一化坐标，值域[0, 1]，坐标分量顺序为[ymin, xmin, ymax, xmax]。reversed_box=false时布局(batch, num, 4)，坐标维为末轴；reversed_box=true时布局(batch, 4, num)，坐标维为次轴。3D张量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape_hw</td>
      <td>输入</td>
      <td>图像尺寸信息，布局为[batch, H, W, C]，对应公式中的H=shape_hw[1]、W=shape_hw[2]。shape固定为(4,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reversed_box</td>
      <td>可选属性</td>
      <td><ul><li>控制bbox数据布局及坐标维位置。false: 布局(batch, num, 4)，坐标维为末轴；true: 布局(batch, 4, num)，坐标维为次轴。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>绝对坐标边界框，对应公式中的绝对坐标。数据类型与shape均与normalized_boxes一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- normalized_boxes与y的数据类型、shape必须一致，仅支持FLOAT16和FLOAT32。
- shape_hw固定为INT32、shape为(4,)，布局为[batch, H, W, C]；H取shape_hw[1]、W取shape_hw[2]。
- normalized_boxes为3D，坐标维（长度为4的轴）的位置由reversed_box决定：false时为末轴，true时为次轴。
- 所有张量仅支持ND格式，不支持空Tensor。
- FLOAT16场景直接在FLOAT16精度下完成坐标缩放，不做FLOAT32中转。
- 确定性说明：本算子为逐元素乘法（无reduce / 无累加），相同输入产生相同输出（bitwise可复现）。

## 调用说明

| 调用方式   | 样例代码                                                                        | 说明                                         |
| ---------------- |-----------------------------------------------------------------------------| --------------------------------------------------- |
| 图模式 | [test_geir_to_absolute_b_box](./examples/arch35/test_geir_to_absolute_b_box.cpp) | 通过[算子IR](./op_graph/to_absolute_b_box_proto.h)构图方式调用 ToAbsoluteBBox 算子。 |
