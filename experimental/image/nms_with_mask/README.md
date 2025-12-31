# NMSWithMask

##  产品支持情况
| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：对候选框执行非极大值抑制（NMS），输出候选框保留掩码，支持 IOU 阈值、置信度阈值过滤。
- 输入坐标格式为 <x1,y1,x2,y2>（左下角和右上角坐标），在代码中以一维数组形式存储（[N,4] 维度展开为长度 N*4 的一维数据）。
- 计算公式：
  - 输入节点：
    - x (shape[N,4], FLOAT32) - 候选框坐标（x1,y1,x2,y2）
    - y (shape[N], FLOAT32)   - 候选框置信度
    - iou_threshold (shape[1], FLOAT32) - IOU过滤阈值
    - scores_threshold (shape[1], FLOAT32) - 置信度过滤阈值

  - 计算节点：
    - Step1: 置信度降序排序索引 idx_sorted ( 该算子默认已降序排序，需用户自行保证)
    - Step2: 初始化临时掩码标记所有候选框为保留状态（值为 1），输出掩码初始化为 0；
    - Step3: 遍历候选框，跳过已标记为过滤的框，过滤置信度低于scores_threshold的框（临时掩码置 0）；
    - Step4: 对保留的候选框标记输出掩码为 1，计算其与后续所有候选框的 IOU（IOU = 交集面积 /(当前框面积 + 对比框面积 - 交集面积)）；
    - Step5：将 IOU 高于iou_threshold的后续框标记为过滤（临时掩码置 0）；
    - 重复步骤 Step3-Step5 直至所有框处理完成。

  - 输出节点：
    - z (shape[N], UINT8) - 候选框保留掩码

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
      <td>x</td>
      <td>输入</td>
      <td>候选框坐标（左下和右上）[N,4]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>置信度。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>输入</td>
      <td>IOU阈值。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scores_threshold</td>
      <td>输入</td>
      <td>置信度阈值。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>选中候选框对应掩码。</td>
      <td>UINT_8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 目前只支持float32输入


## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_nms_with_mask_example](./examples/test_aclnn_nms_with_mask.cpp) | 通过aclnnNMSWithMask接口方式调用NMSWithMask算子。 |