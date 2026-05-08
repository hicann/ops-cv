/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aipp_tiling.h
 * \brief
 */

#ifndef AIPP_OP_HOST_ARCH35_AIPP_TILING_H
#define AIPP_OP_HOST_ARCH35_AIPP_TILING_H

#include <cstdint>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_key.h"
#include "op_host/tiling_util.h"
#include "../../op_kernel/arch35/aipp_struct.h"

namespace optiling {
using namespace std;
using namespace ge;
using namespace Aipp_Kernel;

constexpr int64_t MAX_RGB_BOUND = 255;
constexpr int64_t MIN_MATRIX_BOUND = -32677;
constexpr int64_t MAX_MATRIX_BOUND = 32676;
constexpr int16_t CSC_IDENTITY_SCALE = 256;
constexpr int64_t MIN_CHN_BOUND = -65504;
constexpr int64_t MAX_CHN_BOUND = 65504;
constexpr int64_t MAX_IMAGE_HIGH = 4096;
constexpr int64_t MAX_IMAGE_WIDTH = 4096;

constexpr size_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_THREAD_NUM = 512;
constexpr int64_t DCACHE_SIZE = 1024 * 32;
constexpr uint8_t EVEN_NUMBER_BASE = 2;

constexpr uint8_t CONST_VALUE_ZERO = 0;
constexpr uint8_t CONST_VALUE_ONE = 1;
constexpr uint8_t CONST_VALUE_TWO = 2;
constexpr uint8_t CONST_VALUE_THREE = 3;
constexpr uint8_t CONST_VALUE_FOUR = 4;

constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;

constexpr int64_t AIPP_ERROR_TILINGKEY = 0;
constexpr int64_t AIPP_RGB_PASS_THROUGH = 1;
constexpr int64_t AIPP_YUV_PASS_THROUGH = 2;
constexpr int64_t AIPP_RGB_TO_YUV = 3;
constexpr int64_t AIPP_RGB_TO_GRAY = 4;
constexpr int64_t AIPP_YUV_TO_RGB = 5;
constexpr int64_t AIPP_YUV_TO_GRAY = 6;
constexpr int64_t AIPP_DYNAMIC_DEFAULT = 100;

constexpr size_t INPUT_IMAGES_IDX = 0;
constexpr size_t INPUT_PARAMS_IDX = 1;
constexpr size_t OUTPUT_FEATURES_IDX = 0;
constexpr size_t ATTR_AIPP_CONFIG_PATH_IDX = 0;
constexpr size_t IMAGE_BATCH_DIM = 0;
constexpr size_t NHWC_IMAGE_H_DIM = 1;
constexpr size_t NHWC_IMAGE_W_DIM = 2;
constexpr size_t NHWC_IMAGE_CHANNEL_DIM = 3;
constexpr size_t NCHW_IMAGE_H_DIM = 2;
constexpr size_t NCHW_IMAGE_W_DIM = 3;

constexpr int64_t MAX_PADDING_SIZE = 32;
constexpr int64_t MIN_PADDING_VALUE_UINT8 = 0;
constexpr int64_t MAX_PADDING_VALUE_UINT8 = 255;
constexpr float MIN_PADDING_VALUE_FP16 = (-65504.0f);
constexpr float MAX_PADDING_VALUE_FP16 = (65504.0f);
constexpr int64_t MAX_PADDING_OUT_W = 1080;

const string IMAGE_FORMAT_RGB888_U8 = "RGB888_U8";
const string IMAGE_FORMAT_YUV420SP_U8 = "YUV420SP_U8";
const string IMAGE_FORMAT_YUV400_U8 = "YUV400_U8";
const string IMAGE_FORMAT_XRGB8888_U8 = "XRGB8888_U8";
constexpr uint8_t IMAGE_FORMAT_RGB888_U8_SIZE_LIMIT = (3);
constexpr float IMAGE_FORMAT_YUV420SP_U8_SIZE_LIMIT = (1.5f);
constexpr uint8_t IMAGE_FORMAT_YUV400_U8_SIZE_LIMIT = (1);
constexpr uint8_t IMAGE_FORMAT_XRGB8888_U8_SIZE_LIMIT = (4);

const string AIPP_MODE = "aipp_mode";
const string AIPP_MODE_STATIC = "static";
const string AIPP_MODE_DYNAMIC = "dynamic";
const string AIPP_INPUT_FORMAT = "input_format";
const string AIPP_SRC_IMAGE_SIZE_W = "src_image_size_w";
const string AIPP_SRC_IMAGE_SIZE_H = "src_image_size_h";
const string AIPP_RBUV_SWAP_SWITCH = "rbuv_swap_switch";
const string AIPP_AX_SWAP_SWITCH = "ax_swap_switch";
const string AIPP_CROP = "crop";
const string AIPP_CROP_SIZE_W = "crop_size_w";
const string AIPP_CROP_SIZE_H = "crop_size_h";
const string AIPP_CROP_LOAD_START_POS_H = "load_start_pos_h";
const string AIPP_CROP_LOAD_START_POS_W = "load_start_pos_w";

const string AIPP_CFG_CSC_SWITCH = "csc_switch";
const string AIPP_CFG_MATRIX_R0C0 = "matrix_r0c0";
const string AIPP_CFG_MATRIX_R0C1 = "matrix_r0c1";
const string AIPP_CFG_MATRIX_R0C2 = "matrix_r0c2";
const string AIPP_CFG_MATRIX_R1C0 = "matrix_r1c0";
const string AIPP_CFG_MATRIX_R1C1 = "matrix_r1c1";
const string AIPP_CFG_MATRIX_R1C2 = "matrix_r1c2";
const string AIPP_CFG_MATRIX_R2C0 = "matrix_r2c0";
const string AIPP_CFG_MATRIX_R2C1 = "matrix_r2c1";
const string AIPP_CFG_MATRIX_R2C2 = "matrix_r2c2";
const string AIPP_CFG_OUTPUT_BIAS_0 = "output_bias_0";
const string AIPP_CFG_OUTPUT_BIAS_1 = "output_bias_1";
const string AIPP_CFG_OUTPUT_BIAS_2 = "output_bias_2";
const string AIPP_CFG_INPUT_BIAS_0 = "input_bias_0";
const string AIPP_CFG_INPUT_BIAS_1 = "input_bias_1";
const string AIPP_CFG_INPUT_BIAS_2 = "input_bias_2";

const string AIPP_MEAN_CHN_0 = "mean_chn_0";
const string AIPP_MEAN_CHN_1 = "mean_chn_1";
const string AIPP_MEAN_CHN_2 = "mean_chn_2";
const string AIPP_MEAN_CHN_3 = "mean_chn_3";
const string AIPP_MIN_CHN_0 = "min_chn_0";
const string AIPP_MIN_CHN_1 = "min_chn_1";
const string AIPP_MIN_CHN_2 = "min_chn_2";
const string AIPP_MIN_CHN_3 = "min_chn_3";
const string AIPP_VAR_RECI_CHN_0 = "var_reci_chn_0";
const string AIPP_VAR_RECI_CHN_1 = "var_reci_chn_1";
const string AIPP_VAR_RECI_CHN_2 = "var_reci_chn_2";
const string AIPP_VAR_RECI_CHN_3 = "var_reci_chn_3";

const string AIPP_PADDING = "padding";
const string AIPP_LEFT_PADDING_SIZE = "left_padding_size";
const string AIPP_RIGHT_PADDING_SIZE = "right_padding_size";
const string AIPP_TOP_PADDING_SIZE = "top_padding_size";
const string AIPP_BOTTOM_PADDING_SIZE = "bottom_padding_size";
const string AIPP_PADDING_VALUE = "padding_value";

const std::map<string, uint8_t> IMAGE_FORMAT_MAP = {{IMAGE_FORMAT_YUV420SP_U8, 1},
                                                    {IMAGE_FORMAT_XRGB8888_U8, 2},
                                                    {IMAGE_FORMAT_RGB888_U8, 5},
                                                    {IMAGE_FORMAT_YUV400_U8, 10}};

struct AippCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class AippTiling : public Ops::Cv::OpTiling::TilingBaseClass {
public:
    explicit AippTiling(gert::TilingContext* context) : Ops::Cv::OpTiling::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckAippCfg();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputFormat();
    ge::graphStatus SetImagesValue();
    ge::graphStatus CheckInputImage();
    ge::graphStatus SetCropValue();
    ge::graphStatus CheckCropSize();
    ge::graphStatus SetCscValue();
    ge::graphStatus SetDTCValue();
    ge::graphStatus SetSwapSwitch();
    void SetCscDefaultMatrix();
    void SetCscFormatBias();
    void SetGrayFlag();
    void SwapChannelForCSC();
    ge::graphStatus SetPaddingValue();
    ge::graphStatus CheckPaddingSize();
    ge::graphStatus ValidPaddingValue(float padValue, ge::DataType outputDtype);
    ge::graphStatus ProcessStaticMode();

    map<string, string> parseAippConfig(string jsonStr);
    map<string, string> parseAippCfgFromPath(string fileName);
    void SetSySTilingData();
    void PrintTilingData() const;

    template <typename T>
    ge::graphStatus ParseNumAndValidateRange(const std::map<std::string, T*>& valueMap,\
        int64_t minValue, int64_t maxValue);

private:
    map<string, string> aippCfg;
    AippTilingData tilingData;

    int32_t inputImageSize = 1;
    bool isGray = false;
};

} // namespace optiling
#endif // AIPP_OP_HOST_ARCH35_AIPP_TILING_H