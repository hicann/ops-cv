# Aipp

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

    AIPP（Artificial Intelligence Pre-Processing）人工智能预处理，用于在AI Core上完成数据预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据预处理之后再进行真正的模型推理。**目前仅支持使能静态AIPP功能**，全量的AIPP功能介绍请参考 [《ATC离线模型编译工具》](https://hiascend.com/document/redirect/CannCommunityAtc
      )中“高级功能 > AIPP使能”章节。

- 计算流程：
    1. 根据传入的裁剪参数计算输入和输出的对应关系，对图片进行裁剪。

    2. 根据传入的图片格式和转换参数，对图片进行色域转换，计算公式如下：

        - YUV 转 RGB：

            ```
            | R |   | matrix_r0c0 matrix_r0c1 matrix_r0c2 | | Y - input_bias_0 |
            | G | = | matrix_r1c0 matrix_r1c1 matrix_r1c2 | | U - input_bias_1 | >> 8
            | B |   | matrix_r2c0 matrix_r2c1 matrix_r2c2 | | V - input_bias_2 |
            ```

        - RGB 转 YUV：

            ```
            | Y |   | matrix_r0c0 matrix_r0c1 matrix_r0c2 | | R |        | output_bias_0 |
            | U | = | matrix_r1c0 matrix_r1c1 matrix_r1c2 | | G | >> 8 + | output_bias_1 |
            | V |   | matrix_r2c0 matrix_r2c1 matrix_r2c2 | | B |        | output_bias_2 |
            ```

    3. 对数据进行归一化，计算公式如下:

        ```
        pixel_out_chx(i) = [pixel_in_chx(i) - mean_chn_i - min_chn_i] * var_reci_chn_i
        ```

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
      <td>原始图片输入。</td>
      <td>UINT8</td>
      <td>NHWC</td>
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
      <td>存放配置参数的文件地址，默认值为"./aipp.cfg"。该文件配置方法参见<a href='#约束说明'>约束说明</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输出</td>
      <td>处理后的图片。
      </td>
      <td>FLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- aipp_config_path配置文件说明：仅支持静态AIPP配置、图片裁剪参数配置（Crop）、色域转换参数配置（CSC）、输出类型转换参数配置(DTC)。同时输入图片仅支持三通道。

- aipp_config_path文件配置示例如下，具体参数说明参见下表。

    ```Cpp
    aipp_op {
        aipp_mode : static
        src_image_size_w : 608
        src_image_size_h : 608
        crop : false
        input_format : YUV420SP_U8

        csc_switch : true
        matrix_r0c0 : 298
        matrix_r0c1 : 0
        matrix_r0c2 : 409
        matrix_r1c0 : 298
        matrix_r1c1 : -100
        matrix_r1c2 : -208
        matrix_r2c0 : 298
        matrix_r2c1 : 516
        matrix_r2c2 : 0
        input_bias_0 : 16
        input_bias_1 : 128
        input_bias_2 : 128
    }
    ```

    <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 350px">
    <col style="width: 120px">
    </colgroup>
    <thead>
        <tr >
        <th colspan="2"  style="text-align: center">参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td rowspan="1">全局设置</td>
        <td>aipp_mode</td>
        <td>必选属性</td>
        <td>AIPP算子运行模式，dynamic表示动态AIPP，static表示静态AIPP，目前只支持静态AIPP。</td>
        <td>ENUM</td>
        </tr>
        <tr>
        <td rowspan="3">静态AIPP参数设置</td>
        <td>input_format</td>
        <td>必选属性</td>
        <td>输入图片类型，目前只支持RGB888_U8、YUV420SP_U8。</td>
        <td>ENUM</td>
        </tr>
        <tr>
        <td>src_image_size_h</td>
        <td>可选属性</td>
        <td>输入图像高度，取值范围：[1,4096]；当输入图片类型为YUV420SP_U8时，该值需为偶数。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>src_image_size_w</td>
        <td>可选属性</td>
        <td>输入图像宽度，取值范围：[1,4096]；当输入图片类型为YUV420SP_U8时，该值需为偶数。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td rowspan="5">CROP参数设置</td>
        <td>crop</td>
        <td>可选属性</td>
        <td>裁剪使能开关，true表示支持，false表示不支持。</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>load_start_pos_h</td>
        <td>可选属性</td>
        <td>抠图起始位置垂直方向坐标，取值范围：[0,4095]；当输入图片类型为YUV420SP_U8时，该值需为偶数。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>load_start_pos_w</td>
        <td>可选属性</td>
        <td>抠图起始位置水平方向坐标，取值范围：[0,4095]；当输入图片类型为YUV420SP_U8时，该值需为偶数。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>crop_size_h</td>
        <td>可选属性</td>
        <td>抠图后图像高度，取值范围：[1,4096]。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>crop_size_w</td>
        <td>可选属性</td>
        <td>抠图后图像宽度，取值范围：[1,4096]。</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td rowspan="4">CSC 参数设置</td>
        <td>csc_switch</td>
        <td>可选属性</td>
        <td>色域转换开关，静态AIPP配置，true表示开启色域转换，false表示关闭。</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>
        matrix_r0c0 matrix_r0c1 matrix_r0c2
        matrix_r1c0 matrix_r1c1 matrix_r1c2
        matrix_r2c0 matrix_r2c1 matrix_r2c2
        </td>
        <td>可选属性</td>
        <td>3*3 CSC矩阵元素，取值范围：[-32677 ,32676]。</td>
        <td>INT16</td>
        </tr>
        <tr>
        <td>output_bias_0 output_bias_1 output_bias_2</td>
        <td>可选属性</td>
        <td>RGB转YUV时的输出偏移，取值范围：[0, 255]。</td>
        <td>UINT8</td>
        </tr>
        <tr>
        <td>input_bias_0 input_bias_1 input_bias_2</td>
        <td>可选属性</td>
        <td>YUV转RGB时的输入偏移，取值范围：[0, 255]。</td>
        <td>UINT8 </td>
        </tr>
        <tr>
        <td rowspan="3">DTC参数设置</td>
        <td>mean_chn_0 mean_chn_1 mean_chn_2</td>
        <td>可选属性</td>
        <td>每个通道的均值，取值范围：[0, 255]。</td>
        <td>UINT8</td>
        </tr>
        <tr>
        <td>min_chn_0 min_chn_1 min_chn_2</td>
        <td>可选属性</td>
        <td>每个通道的最小值，取值范围：[0, 255]。</td>
        <td>FLOAT16</td>
        </tr>
        <tr>
        <td>var_reci_chn_0 var_reci_chn_1 var_reci_chn_2</td>
        <td>可选属性</td>
        <td>每个通道方差的倒数，取值范围：[-65504, 65504]。</td>
        <td>FLOAT16</td>
        </tr>
    </tbody></table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_aipp](examples/arch35/test_geir_aipp.cpp) | 通过[算子IR](op_graph/aipp_proto.h)构图方式调用Aipp算子。|
