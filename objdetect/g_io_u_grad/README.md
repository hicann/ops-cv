# GIoUGrad

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

- 算子功能：计算Generalized Intersection over Union (GIoU)损失函数的反向传播梯度。给定上游梯度dy、预测框bboxes和真实框gtboxes，计算GIoU损失对bboxes和gtboxes的梯度。

- 计算公式：

  $$
  GIoU = \frac{inter}{union} - \frac{enclose - union}{enclose}
  $$

  反向传播计算dinter、dunion、denclose，并将梯度分配到各坐标维度。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>上游梯度，1D Tensor，shape=(N,)，N为box数量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>输入</td>
      <td>预测框坐标，2D Tensor，shape=(4, N)。trans=true时为 [x,y,w,h] 格式，trans=false时为 [x1,y1,x2,y2] 格式。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>输入</td>
      <td>真实框坐标，2D Tensor，shape=(4, N)。格式同bboxes。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbboxes</td>
      <td>输出</td>
      <td>bboxes的梯度，2D Tensor，shape=(4, N)，格式与输入bboxes一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgtboxes</td>
      <td>输出</td>
      <td>gtboxes的梯度，2D Tensor，shape=(4, N)，格式与输入gtboxes一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>属性</td>
      <td>Bool类型。true表示xywh格式，false表示xyxy格式。默认值：false。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>属性</td>
      <td>Bool类型。false时M=N（一对一），true时M≠N（交叉计算）。默认值：true。当前仅支持is_cross=false。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>String类型。"iou"表示Intersection over Union，"iof"表示Intersection over Foreground。默认值："iou"。当前仅支持mode="iou"。</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 当前仅支持mode="iou"，不支持mode="iof"。
- 当前仅支持is_cross=false（M=N，一对一计算）。
- 不支持空tensor（N=0 时报错）。
- 需要连续tensor（非连续tensor会自动contiguous）。
- float16输入时内部提升至float32计算，保证精度。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_g_io_u_grad.cpp](./examples/test_geir_g_io_u_grad.cpp) | 通过[算子IR](./op_graph/g_io_u_grad_proto.h)构图方式调用GIoUGrad算子 |
