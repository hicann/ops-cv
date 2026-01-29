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
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_key.h"
#include "tiling_base/tiling_util.h"
#include "../../op_kernel/arch35/aipp_struct.h"

using namespace std;
using namespace ge;

namespace optiling {
constexpr int64_t MAX_RGB_BOUND = 255;
constexpr int64_t MIN_MATRIX_BOUND = -32677;
constexpr int64_t MAX_MATRIX_BOUND = 32676;
constexpr int64_t MIN_CHN_BOUND = -65504;
constexpr int64_t MAX_CHN_BOUND = 65504;

constexpr size_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_THREAD_NUM = 512;
constexpr int64_t DCACHE_SIZE = 1024 * 32;
constexpr uint8_t EVEN_NUMBER_BASE = 2;

constexpr uint8_t CONST_VALUE_ZERO = 0;
constexpr uint8_t CONST_VALUE_ONE = 1;

constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;

constexpr size_t INPUT_IMAGES_IDX = 0;
constexpr size_t INPUT_PARAMS_IDX = 1;
constexpr size_t OUTPUT_FEATURES_IDX = 0;
constexpr size_t ATTR_AIPP_CONFIG_PATH_IDX = 0;
constexpr size_t IMAGE_BATCH_DIM = 0;
constexpr size_t IMAGE_CHANNEL_DIM = 1;
constexpr size_t IMAGE_H_DIM = 2;
constexpr size_t IMAGE_W_DIM = 3;

const string IMAGE_FORMAT_RGB888_U8 = "RGB888_U8";
const uint8_t IMAGE_FORMAT_RGB888_U8_SIZE_LIMIT = 3;

const string AIPP_MODE = "aipp_mode";
const string AIPP_MODE_STATIC = "static";
const string AIPP_INPUT_FORMAT = "input_format";
const string AIPP_SRC_IMAGE_SIZE_W = "src_image_size_w";
const string AIPP_SRC_IMAGE_SIZE_H = "src_image_size_h";
const string AIPP_CROP = "crop";
const string AIPP_CROP_SIZE_W = "crop_size_w";
const string AIPP_CROP_SIZE_H = "crop_size_h";
const string AIPP_CROP_LOAD_START_POS_H = "load_start_pos_h";
const string AIPP_CROP_LOAD_START_POS_W = "load_start_pos_w";

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

const std::map<string, uint8_t> IMAGE_FORMART_MAP = {{IMAGE_FORMAT_RGB888_U8, 1}};

struct AippCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct AippImageInput {
    int32_t inputImageSize = 1;
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
    const ge::graphStatus CheckAippCfg();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputFormat();
    ge::graphStatus SetAippDataFromCfg();
    ge::graphStatus CheckInputImage();
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckAippImages();
    ge::graphStatus CheckCropSize();
    ge::graphStatus CheckTilingData();
    ge::graphStatus SetImagesValue();
    map<string, string> parseAippConfig(string jsonStr);
    map<string, string> parseAippCfgFromPath(string fileName);
    void SetAippTilingData();
    void SetCropValue();
    void SetDTCValue();
    void SetSySTilingData();
    void PrintTilingData();
    bool CheckMeanChnValid(const string& name);
    bool CheckMinChnValid(const string& name);
    bool CheckReciValid(const string& name);
    float StringToFloat(string str);

private:
    AippTilingData tilingData;
    AippImageInput aippImageInput;

    map<string, string> aippCfg;

    uint8_t inputFormat = 0;
    uint8_t imageFormat = 0;
    uint32_t inputSizeH = 0;
    uint32_t inputSizeW = 0;
    uint32_t channelNum = 3;
    uint32_t batchNum = 1;

    // crop
    uint8_t cropSwitch = 0;
    uint32_t cropStartPosH = 0;
    uint32_t cropStartPosW = 0;
    uint32_t cropSizeH = 0;
    uint32_t cropSizeW = 0;

    // DTC
    int16_t dtcPixelMeanChn0 = 0;
    int16_t dtcPixelMeanChn1 = 0;
    int16_t dtcPixelMeanChn2 = 0;
    int16_t dtcPixelMeanChn3 = 0;
    float dtcPixelMinChn0 = 0.0;
    float dtcPixelMinChn1 = 0.0;
    float dtcPixelMinChn2 = 0.0;
    float dtcPixelMinChn3 = 0.0;
    float dtcPixelVarReciChn0 = 1.0;
    float dtcPixelVarReciChn1 = 1.0;
    float dtcPixelVarReciChn2 = 1.0;
    float dtcPixelVarReciChn3 = 1.0;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H