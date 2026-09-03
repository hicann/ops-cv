# GridUnnormal

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：GridUnnormal是GridSample链路中的坐标反归一化算子。输入归一化坐标`grid`和同shape的尺寸辅助张量`assist`，输出小数偏移`diff`和整数采样位置`position`。

- 计算公式：对每个元素独立计算：

  ```text
  t        = (grid + 1) * 0.5
  pos_base = align_corners ? t * (assist - 1) : t * assist - 0.5
  position = floor(pos_base)
  diff     = pos_base - floor(pos_base)
  ```

`floor`按向负无穷取整语义执行，不能用向零截断替代。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| `grid` | 输入 | 归一化采样坐标。支持4D静态shape、动态shape和编译期未知rank；rank确定时必须为4且末维为2；支持总元素数为0的空Tensor。 | `float16`、`float32` | ND |
| `assist` | 输入 | 每元素对应的输入尺寸辅助值。必须与`grid`shape、dtype完全一致，不支持广播；rank确定时必须为4且末维为2。 | `float16`、`float32` | ND |
| `diff` | 输出 | `pos_base`的小数部分，shape与`grid`一致，dtype与`grid`一致。 | `float16`、`float32` | ND |
| `position` | 输出 | `pos_base`的下取整结果，shape与`grid`一致。 | `int32` | ND |
| `align_corners` | 属性 | 可选属性，默认`false`。为`true`时按`t * (assist - 1)`计算；为`false`时按`t * assist - 0.5`计算。 | `bool` | - |

## 约束说明

- `grid`与`assist`的shape、dtype必须一致。
- rank确定时，`grid`与`assist`必须为4D Tensor，shape为`[batch, height, width, 2]`。
- 仅支持ND格式；非连续Tensor作为用户可见接口不涉及。
- `diff`的dtype跟随`grid`；`position`固定为int32。
- 中间计算使用fp32；fp16输入会提升到fp32计算，`diff`再回写为fp16。
- 总元素数为0的空Tensor支持空进空出，设备侧不访问数据。
- 非有限输入以及`floor(pos_base)`超出int32表示范围不属于本算子支持域。
- 本算子为GE图内部算子，不提供aclnn、torch、TensorFlow、ONNX、Caffe对外接口。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_grid_unnormal](examples/arch35/test_geir_grid_unnormal.cpp) | 通过本目录的[算子原型](op_graph/grid_unnormal_proto.h)构图方式调用GridUnnormal算子。 |
