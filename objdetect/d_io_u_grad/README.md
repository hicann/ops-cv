# DIoUGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：计算Distance-IoU (DIoU)损失函数的反向梯度。DIoUGrad是目标检测中DIoU Loss的反向传播算子，计算上游梯度dy对预测框bboxes和真实框gtboxes的梯度。

- 计算公式：

  DIoU前向损失：

  $$
  L_{DIoU} = 1 - IoU + \frac{\rho^2}{c^2}
  $$

  其中：
  - $IoU = \frac{|B \cap B_{gt}|}{|B \cup B_{gt}|}$ 为交并比
  - $\rho^2 = (b_{cx} - b_{gt\_cx})^2 + (b_{cy} - b_{gt\_cy})^2$ 为中心点距离平方
  - $c^2 = c_w^2 + c_h^2$ 为最小外接矩形对角线平方

  反向梯度：

  $$
  \frac{\partial L_{DIoU}}{\partial B} = f(dy, IoU, \rho^2, c^2)
  $$

  $$
  \frac{\partial L_{DIoU}}{\partial B_{gt}} = f(dy, IoU, \rho^2, c^2)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 220px">
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
      <td>dy</td>
      <td>输入</td>
      <td>上游梯度，1D Tensor，N为框数量</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>输入</td>
      <td>预测边界框，2D Tensor，shape (4, N)，4 指 [x1,y1,x2,y2] 或 [x,y,w,h]</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>输入</td>
      <td>真实边界框，2D Tensor，shape (4, M)，4 指 [x1,y1,x2,y2] 或 [x,y,w,h]</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbboxes</td>
      <td>输出</td>
      <td>对预测框的梯度，shape (4, N)</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgtboxes</td>
      <td>输出</td>
      <td>对真实框的梯度，shape (4, M)</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>属性</td>
      <td>true: xywh格式; false: xyxy格式。当前仅支持true</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>属性</td>
      <td>false: M=N（一一对应）; true: M≠N（交叉计算）。当前仅支持false</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>"iou": inter/union; "iof": inter/bboxes_area。当前仅支持 "iou"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入的dtype必须一致（dy、bboxes、gtboxes必须同为float16或同为float32）
- dy必须为1D Tensor
- bboxes和gtboxes必须为2D Tensor，且dim[0]必须为 4
- trans属性支持true（xywh）和false（xyxy）两种格式
- is_cross属性当前仅支持false（M=N）
- mode属性当前仅支持"iou"
- 输出dbboxes的shape与bboxes一致，dgtboxes的shape与gtboxes一致
- 支持空tensor（N=0 时输出shape为 (4, 0)），此时输出为空梯度
- 内部计算使用float32精度，fp16输入时内部提升为fp32

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_d_io_u_grad.cpp](./examples/test_geir_d_io_u_grad.cpp) | 通过[算子IR](./op_graph/d_io_u_grad_proto.h)构图方式调用DIoUGrad算子 |
