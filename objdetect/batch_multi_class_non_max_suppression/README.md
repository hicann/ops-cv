# BatchMultiClassNonMaxSuppression

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对每个 batch、每个类别的候选框执行贪心非极大值抑制（NMS），再从所有类别的保留结果中按分数选择最多 `max_total_size` 个检测框。
- boxes 坐标格式为 `[y_min, x_min, y_max, x_max]`。当 `q=1` 时各类别共享 boxes；当 `q=C` 时每个类别使用自己的 boxes。
- 当提供 `clip_window` 时，算子先执行裁剪；`change_coordinate_frame=true` 时，再以窗口左上角为原点、窗口宽高为尺度进行归一化。

对候选框 `a` 和已选框 `b`，IoU 的计算为：

$$
IoU(a,b)=\frac{Area(a\cap b)}{max(Area(a)+Area(b)-Area(a\cap b), 1e-12)}
$$

分数仅在 `score > score_threshold` 时保留；IoU 严格大于 `iou_threshold` 时抑制。每个类别最多保留 `max_size_per_class` 个框，最终结果不足 `max_total_size` 时以 0 填充。

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
      <td>boxes</td>
      <td>输入</td>
      <td>候选框，坐标格式为[y_min, x_min, y_max, x_max]，shape为[B,N,q,4]；q为1或C。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scores</td>
      <td>输入</td>
      <td>每个候选框、每个类别的分数，shape为[B,N,C]，数据类型必须与boxes一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_window</td>
      <td>可选输入</td>
      <td>裁剪窗口，坐标格式为yxyx，shape为[B,4]。可传空指针，数据类型必须与boxes一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_valid_boxes</td>
      <td>可选输入</td>
      <td>每个batch的有效候选框数，shape为[B]。可传空指针。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>属性</td>
      <td>分数阈值，仅保留严格大于该值的候选框；必须为有限值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>属性</td>
      <td>IoU抑制阈值；必须为[0,1]内的有限值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_size_per_class</td>
      <td>属性</td>
      <td>每个类别最多保留的候选框数量，取值范围为[1,1000]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_total_size</td>
      <td>属性</td>
      <td>每个batch最多保留的候选框数量，取值范围为[1,1000]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>change_coordinate_frame</td>
      <td>属性</td>
      <td>是否将裁剪后的坐标归一化到窗口坐标系，默认为false；为true时必须提供clip_window。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_box</td>
      <td>属性</td>
      <td>是否在算子前插入Transpose，当前仅支持false，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nmsed_boxes</td>
      <td>输出</td>
      <td>NMS后的检测框，shape为[B,M,4]，其中M为max_total_size。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_scores</td>
      <td>输出</td>
      <td>NMS后的检测分数，shape为[B,M]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_classes</td>
      <td>输出</td>
      <td>NMS后的类别编号，shape为[B,M]，以浮点类型表示。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_num</td>
      <td>输出</td>
      <td>每个batch的有效输出数量，shape为[B]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持 ND format、float16/float32 的 boxes 和 scores；三个浮点输出与 boxes dtype 保持一致。
- boxes 为 4 维、scores 为 3 维，B、N、C、q 必须为正；q 必须为 1 或 C。
- `transpose_box` 当前仅支持 `false`，与 910B/910C 的公开接口约束保持一致。
- `clip_window` 的形状必须为 `[B,4]`，`num_valid_boxes` 的形状必须为 `[B]`。
- 当前 Ascend950 tiling 只接受具体的正 shape，不支持动态 rank 和未知维度。
- `image_size` 为图模式原型兼容属性，当前通用 NMS 路径不读取该属性；旧平台 `norm_class` 专用语义不在本实现范围内。
- 相同分数的相对顺序不是公共接口承诺；调用方应以 `nmsed_num` 确定有效输出范围。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式 | [test_geir_batch_multi_class_non_max_suppression](examples/arch35/test_geir_batch_multi_class_non_max_suppression.cpp) | 通过[算子IR](./op_graph/batch_multi_class_non_max_suppression_proto.h)构图方式调用BatchMultiClassNonMaxSuppression算子。 |
