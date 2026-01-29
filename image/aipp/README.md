# GridSample

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：提供Aipp算子的CROP和Data Type Conversion功能。
- 计算公式：

  - 计算流程：
    1. 根据传入的裁剪参数计算输入和输出的对应关系，对图片进行裁剪。
    2. 对数据进行归一化，计算公式如下。dst = (static_cast\<T\>(src - dtcParam.meanChn) - static_cast\<T\>(dtcParam.minChn)) * static_cast\<T\>(dtcParam.varReciChn)

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
      <td>images</td>
      <td>输入</td>
      <td>原始图片，支持uint8数据类型，NCHW和NHWC Format。</td>
      <td>UINT8</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>params</td>
      <td>输入</td>
      <td>可选参数，当前版本用不上，预留。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>aipp_config_path</td>
      <td>可选属性</td>
      <td><ul><li>存放计算参数的文件地址。</li><li>默认值为"./aipp.cfg"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输出</td>
      <td>输出处理后的图片，当前仅支持FLOAT16格式，Format和输入相同，支持NCHW和NHWC。
      </td>
      <td>FLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
  </tbody></table>
- aipp.cfg文件内容。
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
      <td>aipp_mode</td>
      <td>可选属性</td>
      <td>动态还是静态</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_format</td>
      <td>可选属性</td>
      <td>图片类型（仅支持RGB888_U8）</td>
      <td>UINT8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>src_image_size_w</td>
      <td>可选属性</td>
      <td>输入图片W轴大小
      </td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>src_image_size_h</td>
      <td>可选属性</td>
      <td>输入图片H轴大小</td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>crop</td>
      <td>可选属性</td>
      <td>裁剪使能开关</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>load_start_pos_h</td>
      <td>可选属性</td>
      <td>H轴裁剪起始位置</td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>load_start_pos_w</td>
      <td>可选属性</td>
      <td>W轴裁剪起始位置
      </td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>crop_size_h</td>
      <td>可选属性</td>
      <td>裁剪后H轴大小</td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>crop_size_w</td>
      <td>可选属性</td>
      <td>裁剪后W轴大小</td>
      <td>UINT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>csc相关参数</td>
      <td>可选属性</td>
      <td>csc参数暂不使用，省略</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean_chn_0，mean_chn_1，mean_chn_2</td>
      <td>可选属性</td>
      <td>通道均值
      </td>
      <td>INT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>min_chn_0，min_chn_1，min_chn_2</td>
      <td>可选属性</td>
      <td>通道最小值</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var_reci_chn_0，var_reci_chn_1，var_reci_chn_2</td>
      <td>可选属性</td>
      <td>通道方差的倒数</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
  </tbody></table>
## 约束说明

- 使能方式
  - 仅支持静态图调用。
- 约束限制
  - Dtype:原始输入数据支持UINT8格式，输出数据支持FLOAT16格式。
  - Format：输出format和输入相同，支持NHWC和NCHW。
  - soc:Ascend950
  - Aipp参数：Aipp参数：仅支持Crop参数配置、输出类型转换参数配置。仅支持三通道。



## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/aipp_proto.h)构图方式调用GridSample算子。         |
