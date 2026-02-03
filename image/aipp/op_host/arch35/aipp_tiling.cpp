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
 * \file aipp_tiling.cpp
 * \brief
 */

#include <stdexcept>
#include <typeinfo>
#include "aipp_tiling.h"

namespace optiling {
uint64_t AippTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (tilingData.imageFormat == IMAGE_FORMART_MAP.at(IMAGE_FORMAT_RGB888_U8) && tilingData.cscSwitch == 0) {
        tilingKey = FORMAT_RGB_INDICE_UINT32;
    }
    if (tilingData.imageFormat == IMAGE_FORMART_MAP.at(IMAGE_FORMAT_RGB888_U8) && tilingData.cscSwitch == 1) {
        tilingKey = FORMAT_RGB_SWITCH_OPEN_UINT32;
    }
    if (tilingKey == 0) {
        OP_LOGE(context_->GetNodeName(), "tilingKey is:0, please check aipp.cfg's inputformat and csc_switch");
    }
    return tilingKey;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static bool isJsonSymbol(char c)
{
    const std::string symbols = "\"{}[](),:;";
    return symbols.find(c) != std::string::npos;
}

ge::graphStatus AippTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo begin.");
    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputDtype fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputFormat() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputFormat fail."),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* configData = attrs->GetAttrPointer<char>(ATTR_AIPP_CONFIG_PATH_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, configData);

    string configStr(configData);
    if (!configStr.empty() && configStr.front() == '{' && configStr.back() == '}') {
        OP_LOGI(context_->GetNodeName(), "aippConfigData is: %s", configData);
        aippCfg = parseAippConfig(configData);
    } else {
        OP_LOGI(context_->GetNodeName(), "aippConfigPath is: %s", configData);
        aippCfg = parseAippCfgFromPath(configData);
    }

    OP_CHECK_IF(
        CheckAippCfg() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "aipp.cfg is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckAippImages() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckAippImages fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetAippDataFromCfg() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "aipp.cfg's value is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputFormat()
{
    auto inputImges = context_->GetInputDesc(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputImges);
    auto inputFormatLocal = inputImges->GetOriginFormat();
    OP_CHECK_IF(
        inputFormatLocal != ge::FORMAT_NCHW && inputFormatLocal != ge::FORMAT_NHWC,
        OP_LOGE(context_->GetNodeName(), "aipp input format only support NCHW and NHWC."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus AippTiling::CheckAippCfg()
{
    OP_CHECK_IF(aippCfg.size() == 0, OP_LOGE(context_->GetNodeName(), "aippCfg is empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        aippCfg.find(AIPP_MODE) == aippCfg.end(), OP_LOGE(context_->GetNodeName(), "aippCfg has no aipp_mode."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        aippCfg.at(AIPP_MODE) != AIPP_MODE_STATIC, OP_LOGE(context_->GetNodeName(), "aippCfg value is not static."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::SetAippDataFromCfg()
{
    SetCropValue();
    if (CheckCropSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetCscValue();
    SetDTCValue();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::SetImagesValue()
{
    if (aippCfg.find(AIPP_INPUT_FORMAT) != aippCfg.end()) {
        OP_CHECK_IF(
            aippCfg.at(AIPP_INPUT_FORMAT) != IMAGE_FORMAT_RGB888_U8,
            OP_LOGE(context_->GetNodeName(), "aipp input_format only support RGB888_U8."), return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            IMAGE_FORMART_MAP.find(aippCfg.at(AIPP_INPUT_FORMAT)) == IMAGE_FORMART_MAP.end(),
            OP_LOGE(context_->GetNodeName(), "can not find RGB888_U8 in IMAGE_FORMART_MAP."), return ge::GRAPH_FAILED);

        imageFormat = IMAGE_FORMART_MAP.at(aippCfg.at(AIPP_INPUT_FORMAT));
    }
    if (aippCfg.find(AIPP_SRC_IMAGE_SIZE_H) != aippCfg.end()) {
        uint32_t srcImageSizeH = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_SRC_IMAGE_SIZE_H)));
        if (srcImageSizeH > 0) {
            inputSizeH = srcImageSizeH;
        }
    }
    if (aippCfg.find(AIPP_SRC_IMAGE_SIZE_W) != aippCfg.end()) {
        uint32_t srcImageSizeW = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_SRC_IMAGE_SIZE_W)));
        if (srcImageSizeW > 0) {
            inputSizeW = srcImageSizeW;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputImage()
{
    if (aippImageInput.inputImageSize < batchNum * inputSizeW * inputSizeH * IMAGE_FORMAT_RGB888_U8_SIZE_LIMIT) {
        stringstream errorCheckLog;
        errorCheckLog << "when input_format is RGB888_U8, input image size "
                      << "should be bigger than N * src_image_size_w * src_image_size_h * 3.";
        OP_LOGE(context_->GetNodeName(), "%s", errorCheckLog.str().c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputDtype()
{
    auto inputImges = context_->GetInputDesc(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputImges);
    OP_CHECK_IF(
        inputImges->GetDataType() != ge::DT_UINT8, OP_LOGE(context_->GetNodeName(), "inputImges only support uint8."),
        return ge::GRAPH_FAILED);
    auto outputImges = context_->GetOutputDesc(OUTPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputImges);
    OP_CHECK_IF(
        outputImges->GetDataType() != ge::DT_FLOAT16,
        OP_LOGE(context_->GetNodeName(), "outputImges only support float16."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

map<string, string> AippTiling::parseAippCfgFromPath(string fileName)
{
    map<string, string> aippConfig;
    if (fileName.find("..") != string::npos) {
        OP_LOGE(context_->GetNodeName(), "fileName: %s has invalid char.", fileName.c_str());
        return aippConfig;
    }
    ifstream file(fileName);
    if (!file.is_open()) {
        OP_LOGE(context_->GetNodeName(), "Failed to open file: %s", fileName.c_str());
        return aippConfig;
    }
    string line;
    string currentSection;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.find('{') != string::npos) {
            size_t start = line.find_first_not_of(" \t");
            if (start != string::npos) {
                currentSection = line.substr(start, line.find('{') - start);
            }
            continue;
        }
        if (line.find('}') != string::npos) {
            continue;
        }
        size_t colonPos = line.find(':');
        if (colonPos != string::npos) {
            string key = line.substr(0, colonPos);
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
            string value = line.substr(colonPos + 1);
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            aippConfig[key] = value;
        }
    }
    file.close();
    return aippConfig;
}

map<string, string> AippTiling::parseAippConfig(string jsonStr)
{
    map<string, string> aippConfig;
    replace(jsonStr.begin(), jsonStr.end(), ',', '\n');

    istringstream file(jsonStr);
    string line;
    while (getline(file, line)) {
        size_t colonPos = line.find(':');
        if (colonPos != string::npos) {
            string key = line.substr(0, colonPos);
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
            key.erase(std::remove_if(key.begin(), key.end(), isJsonSymbol), key.end());
            string value = line.substr(colonPos + 1);
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            value.erase(std::remove_if(value.begin(), value.end(), isJsonSymbol), value.end());
            aippConfig[key] = value;
        }
    }
    return aippConfig;
}

ge::graphStatus AippTiling::CheckAippImages()
{
    auto images = context_->GetInputShape(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, images);
    auto imagesShape = images->GetStorageShape();

    auto oriInputFormat = context_->GetInputDesc(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, oriInputFormat);
    if (oriInputFormat->GetOriginFormat() == ge::FORMAT_NCHW) {
        inputFormat = NCHW_FORMAT_INDEX;
        batchNum = imagesShape.GetDim(IMAGE_BATCH_DIM);
        channelNum = imagesShape.GetDim(IMAGE_CHANNEL_DIM);
        inputSizeH = imagesShape.GetDim(IMAGE_H_DIM);
        inputSizeW = imagesShape.GetDim(IMAGE_W_DIM);
    } else {
        inputFormat = NHWC_FORMAT_INDEX;
        batchNum = imagesShape.GetDim(IMAGE_BATCH_DIM);
        inputSizeH = imagesShape.GetDim(IMAGE_CHANNEL_DIM);
        inputSizeW = imagesShape.GetDim(IMAGE_H_DIM);
        channelNum = imagesShape.GetDim(IMAGE_W_DIM);
    }

    aippImageInput.inputImageSize = imagesShape.GetShapeSize();

    if (SetImagesValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputImage() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool AippTiling::CheckMatrixValid(const string& name)
{
    if (aippCfg.find(name) == aippCfg.end()) {
        return false;
    }
    if (name == AIPP_CFG_MATRIX_R0C0) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C0)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C0)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix00 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R0C1) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C1)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C1)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix01 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R0C2) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C2)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C2)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix02 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R1C0) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C0)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C0)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix10 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R1C1) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C1)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C1)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix11 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R1C2) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C2)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C2)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix12 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R2C0) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C0)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C0)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix20 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R2C1) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C1)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C1)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix21 should between [-32677,32676]."), return false);
    } else if (name == AIPP_CFG_MATRIX_R2C2) {
        OP_CHECK_IF(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C2)) < MIN_MATRIX_BOUND ||
            stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C2)) > MAX_MATRIX_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of cscMatrix22 should between [-32677,32676]."), return false);
    }
    return true;
}

bool AippTiling::CheckBaisValid(const string& name)
{
    if (aippCfg.find(name) == aippCfg.end()) {
        return false;
    }
    if (name == AIPP_CFG_INPUT_BIAS_0) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_0)) < 0 || stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_0)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of inBias0 should between [0,255]."), return false);
    } else if (name == AIPP_CFG_INPUT_BIAS_1) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_1)) < 0 || stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_1)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of inBias1 should between [0,255]."), return false);
    } else if (name == AIPP_CFG_INPUT_BIAS_2) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_2)) < 0 || stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_2)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of inBias2 should between [0,255]."), return false);
    } else if (name == AIPP_CFG_OUTPUT_BIAS_0) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_0)) < 0 || stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_0)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of outBias0 should between [0,255]."), return false);
    } else if (name == AIPP_CFG_OUTPUT_BIAS_1) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_1)) < 0 || stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_1)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of outBias1 should between [0,255]."), return false);
    } else if (name == AIPP_CFG_OUTPUT_BIAS_2) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_2)) < 0 || stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_2)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of outBias2 should between [0,255]."), return false);
    }
    return true;
}

bool AippTiling::CheckMeanChnValid(const string& name)
{
    if (aippCfg.find(name) == aippCfg.end()) {
        return false;
    }
    if (name == AIPP_MEAN_CHN_0) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MEAN_CHN_0)) < 0 || stoi(aippCfg.at(AIPP_MEAN_CHN_0)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMeanChn0 should between [0,255]."), return false);
    } else if (name == AIPP_MEAN_CHN_1) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MEAN_CHN_1)) < 0 || stoi(aippCfg.at(AIPP_MEAN_CHN_1)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMeanChn1 should between [0,255]."), return false);
    } else if (name == AIPP_MEAN_CHN_2) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MEAN_CHN_2)) < 0 || stoi(aippCfg.at(AIPP_MEAN_CHN_2)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMeanChn2 should between [0,255]."), return false);
    } else if (name == AIPP_MEAN_CHN_3) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MEAN_CHN_3)) < 0 || stoi(aippCfg.at(AIPP_MEAN_CHN_3)) > MAX_RGB_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMeanChn3 should between [0,255]."), return false);
    }
    return true;
}

bool AippTiling::CheckMinChnValid(const string& name)
{
    if (aippCfg.find(name) == aippCfg.end()) {
        return false;
    }
    if (name == AIPP_MIN_CHN_0) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MIN_CHN_0)) < MIN_CHN_BOUND || stoi(aippCfg.at(AIPP_MIN_CHN_0)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMinChn0 should between [-65504,65504]."), return false);
    } else if (name == AIPP_MIN_CHN_1) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MIN_CHN_1)) < MIN_CHN_BOUND || stoi(aippCfg.at(AIPP_MIN_CHN_1)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMinChn1 should between [-65504,65504]."), return false);
    } else if (name == AIPP_MIN_CHN_2) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MIN_CHN_2)) < MIN_CHN_BOUND || stoi(aippCfg.at(AIPP_MIN_CHN_2)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMinChn2 should between [-65504,65504]."), return false);
    } else if (name == AIPP_MIN_CHN_3) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_MIN_CHN_3)) < MIN_CHN_BOUND || stoi(aippCfg.at(AIPP_MIN_CHN_3)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelMinChn3 should between [-65504,65504]."), return false);
    }
    return true;
}

bool AippTiling::CheckReciValid(const string& name)
{
    if (aippCfg.find(name) == aippCfg.end()) {
        return false;
    }
    if (name == AIPP_VAR_RECI_CHN_0) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_VAR_RECI_CHN_0)) < MIN_CHN_BOUND ||
                stoi(aippCfg.at(AIPP_VAR_RECI_CHN_0)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelReciChn0 should between [-65504,65504]."),
            return false);
    } else if (name == AIPP_VAR_RECI_CHN_1) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_VAR_RECI_CHN_1)) < MIN_CHN_BOUND ||
                stoi(aippCfg.at(AIPP_VAR_RECI_CHN_1)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelReciChn1 should between [-65504,65504]."),
            return false);
    } else if (name == AIPP_VAR_RECI_CHN_2) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_VAR_RECI_CHN_2)) < MIN_CHN_BOUND ||
                stoi(aippCfg.at(AIPP_VAR_RECI_CHN_2)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelReciChn2 should between [-65504,65504]."),
            return false);
    } else if (name == AIPP_VAR_RECI_CHN_3) {
        OP_CHECK_IF(
            stoi(aippCfg.at(AIPP_VAR_RECI_CHN_3)) < MIN_CHN_BOUND ||
                stoi(aippCfg.at(AIPP_VAR_RECI_CHN_3)) > MAX_CHN_BOUND,
            OP_LOGE(context_->GetNodeName(), "Params of dtcPixelReciChn3 should between [-65504,65504]."),
            return false);
    }
    return true;
}

ge::graphStatus AippTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AippTiling::IsCapable()
{
    return true;
}

ge::graphStatus AippTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYSTEM_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::DoOpTiling()
{
    SetAippTilingData();
    OP_CHECK_IF(
        (CheckTilingData() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "Check aipp params is invalid."),
        return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AippTiling::SetAippTilingData()
{
    tilingData.inputFormat = inputFormat;
    tilingData.imageFormat = imageFormat;
    tilingData.batchNum = batchNum;
    tilingData.channelNum = channelNum;
    tilingData.inputSizeH = inputSizeH;
    tilingData.inputSizeW = inputSizeW;

    tilingData.cropSwitch = cropSwitch;
    tilingData.cropStartPosH = cropStartPosH;
    tilingData.cropStartPosW = cropStartPosW;
    tilingData.cropSizeH = cropSizeH;
    tilingData.cropSizeW = cropSizeW;

    tilingData.dtcPixelMeanChn0 = dtcPixelMeanChn0;
    tilingData.dtcPixelMeanChn1 = dtcPixelMeanChn1;
    tilingData.dtcPixelMeanChn2 = dtcPixelMeanChn2;
    tilingData.dtcPixelMeanChn3 = dtcPixelMeanChn3;
    tilingData.dtcPixelMinChn0 = dtcPixelMinChn0;
    tilingData.dtcPixelMinChn1 = dtcPixelMinChn1;
    tilingData.dtcPixelMinChn2 = dtcPixelMinChn2;
    tilingData.dtcPixelMinChn3 = dtcPixelMinChn3;
    tilingData.dtcPixelVarReciChn0 = dtcPixelVarReciChn0;
    tilingData.dtcPixelVarReciChn1 = dtcPixelVarReciChn1;
    tilingData.dtcPixelVarReciChn2 = dtcPixelVarReciChn2;
    tilingData.dtcPixelVarReciChn3 = dtcPixelVarReciChn3;

    tilingData.cscSwitch = cscSwitch;
    tilingData.cscMatrix00 = cscMatrix00;
    tilingData.cscMatrix01 = cscMatrix01;
    tilingData.cscMatrix02 = cscMatrix02;
    tilingData.cscMatrix10 = cscMatrix10;
    tilingData.cscMatrix11 = cscMatrix11;
    tilingData.cscMatrix12 = cscMatrix12;
    tilingData.cscMatrix20 = cscMatrix20;
    tilingData.cscMatrix21 = cscMatrix21;
    tilingData.cscMatrix22 = cscMatrix22;

    tilingData.outBias0 = outBias0;
    tilingData.outBias1 = outBias1;
    tilingData.outBias2 = outBias2;
    tilingData.inBias0 = inBias0;
    tilingData.inBias1 = inBias1;
    tilingData.inBias2 = inBias2;
}

void AippTiling::SetCropValue()
{
    bool cropEnabled = (aippCfg.find(AIPP_CROP) != aippCfg.end()) && (aippCfg.at(AIPP_CROP) == "true");
    cropSwitch = cropEnabled ? CONST_VALUE_ONE : CONST_VALUE_ZERO;
    if (cropEnabled) {
        if (aippCfg.find(AIPP_CROP_LOAD_START_POS_W) != aippCfg.end()) {
            cropStartPosW = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_CROP_LOAD_START_POS_W)));
        }
        if (aippCfg.find(AIPP_CROP_LOAD_START_POS_H) != aippCfg.end()) {
            cropStartPosH = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_CROP_LOAD_START_POS_H)));
        }
        if (aippCfg.find(AIPP_CROP_SIZE_W) != aippCfg.end()) {
            cropSizeW = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_CROP_SIZE_W)));
        }
        if (aippCfg.find(AIPP_CROP_SIZE_H) != aippCfg.end()) {
            cropSizeH = static_cast<uint32_t>(stoi(aippCfg.at(AIPP_CROP_SIZE_H)));
        }
    } else {
        cropSizeW = inputSizeW;
        cropSizeH = inputSizeH;
    }
}

ge::graphStatus AippTiling::CheckCropSize()
{
    if (cropSwitch == 1) {
        if (cropStartPosW + cropSizeW > inputSizeW) {
            stringstream cropSizeError;
            cropSizeError << "aipp.cfg's crop_start_pos_w + crop_size_w must <= src_iimage_size_w, "
                          << "current cropStartPosW add cropSizeW value is: " << cropStartPosW + cropSizeW;
            OP_LOGE(context_->GetNodeName(), "%s", cropSizeError.str().c_str());
            return ge::GRAPH_FAILED;
        }
        if (cropStartPosH + cropSizeH > inputSizeH) {
            stringstream cropSizeError;
            cropSizeError << "aipp.cfg's crop_start_pos_h + crop_size_h must <= src_iimage_size_h, "
                          << "current cropStartPosH add cropSizeH value is:" << cropStartPosH + cropSizeH;
            OP_LOGE(context_->GetNodeName(), "%s", cropSizeError.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void AippTiling::SetCscValue()
{
    bool cscEnabled = (aippCfg.find(AIPP_CFG_CSC_SWITCH) != aippCfg.end()) &&
                      (aippCfg.at(AIPP_CFG_CSC_SWITCH) == "true");
    cscSwitch = cscEnabled ? CONST_VALUE_ONE : CONST_VALUE_ZERO;
    if (cscEnabled) {
        SetMatrixValid();
        SetBaisValid();
    }
}

void AippTiling::SetMatrixValid()
{
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R0C0)) {
        cscMatrix00 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C0)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R0C1)) {
        cscMatrix01 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C1)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R0C2)) {
        cscMatrix02 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R0C2)));
    }

    if (CheckMatrixValid(AIPP_CFG_MATRIX_R1C0)) {
        cscMatrix10 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C0)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R1C1)) {
        cscMatrix11 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C1)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R1C2)) {
        cscMatrix12 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R1C2)));
    }

    if (CheckMatrixValid(AIPP_CFG_MATRIX_R2C0)) {
        cscMatrix20 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C0)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R2C1)) {
        cscMatrix21 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C1)));
    }
    if (CheckMatrixValid(AIPP_CFG_MATRIX_R2C2)) {
        cscMatrix22 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_CFG_MATRIX_R2C2)));
    }
}

void AippTiling::SetBaisValid()
{
    if (CheckBaisValid(AIPP_CFG_INPUT_BIAS_0)) {
        inBias0 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_0)));
    }
    if (CheckBaisValid(AIPP_CFG_INPUT_BIAS_1)) {
        inBias1 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_1)));
    }
    if (CheckBaisValid(AIPP_CFG_INPUT_BIAS_2)) {
        inBias2 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_INPUT_BIAS_2)));
    }

    if (CheckBaisValid(AIPP_CFG_OUTPUT_BIAS_0)) {
        outBias0 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_0)));
    }
    if (CheckBaisValid(AIPP_CFG_OUTPUT_BIAS_1)) {
        outBias1 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_1)));
    }
    if (CheckBaisValid(AIPP_CFG_OUTPUT_BIAS_2)) {
        outBias2 = static_cast<uint8_t>(stoi(aippCfg.at(AIPP_CFG_OUTPUT_BIAS_2)));
    }
    
    bool hasInBiasFlag = aippCfg.find(AIPP_CFG_INPUT_BIAS_0) != aippCfg.end() &&
                            aippCfg.find(AIPP_CFG_INPUT_BIAS_1) != aippCfg.end() &&
                            aippCfg.find(AIPP_CFG_INPUT_BIAS_2) != aippCfg.end();
    bool hasOutBiasFlag = aippCfg.find(AIPP_CFG_OUTPUT_BIAS_0) != aippCfg.end() &&
                            aippCfg.find(AIPP_CFG_OUTPUT_BIAS_1) != aippCfg.end() &&
                            aippCfg.find(AIPP_CFG_OUTPUT_BIAS_2) != aippCfg.end();
    if (cscSwitch && hasInBiasFlag && !hasOutBiasFlag) {
        outBias0 = inBias0;
        outBias1 = inBias1;
        outBias2 = inBias2;
    }
}

void AippTiling::SetDTCValue()
{
    if (CheckMeanChnValid(AIPP_MEAN_CHN_0)) {
        dtcPixelMeanChn0 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_MEAN_CHN_0)));
    }
    if (CheckMeanChnValid(AIPP_MEAN_CHN_1)) {
        dtcPixelMeanChn1 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_MEAN_CHN_1)));
    }
    if (CheckMeanChnValid(AIPP_MEAN_CHN_2)) {
        dtcPixelMeanChn2 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_MEAN_CHN_2)));
    }
    if (CheckMeanChnValid(AIPP_MEAN_CHN_3)) {
        dtcPixelMeanChn3 = static_cast<int16_t>(stoi(aippCfg.at(AIPP_MEAN_CHN_3)));
    }
    if (CheckMinChnValid(AIPP_MIN_CHN_0)) {
        dtcPixelMinChn0 = StringToFloat(aippCfg.at(AIPP_MIN_CHN_0));
    }
    if (CheckMinChnValid(AIPP_MIN_CHN_1)) {
        dtcPixelMinChn1 = StringToFloat(aippCfg.at(AIPP_MIN_CHN_1));
    }
    if (CheckMinChnValid(AIPP_MIN_CHN_2)) {
        dtcPixelMinChn2 = StringToFloat(aippCfg.at(AIPP_MIN_CHN_2));
    }
    if (CheckMinChnValid(AIPP_MIN_CHN_3)) {
        dtcPixelMinChn3 = StringToFloat(aippCfg.at(AIPP_MIN_CHN_3));
    }
    if (CheckReciValid(AIPP_VAR_RECI_CHN_0)) {
        dtcPixelVarReciChn0 = StringToFloat(aippCfg.at(AIPP_VAR_RECI_CHN_0));
    }
    if (CheckReciValid(AIPP_VAR_RECI_CHN_1)) {
        dtcPixelVarReciChn1 = StringToFloat(aippCfg.at(AIPP_VAR_RECI_CHN_1));
    }
    if (CheckReciValid(AIPP_VAR_RECI_CHN_2)) {
        dtcPixelVarReciChn2 = StringToFloat(aippCfg.at(AIPP_VAR_RECI_CHN_2));
    }
    if (CheckReciValid(AIPP_VAR_RECI_CHN_3)) {
        dtcPixelVarReciChn3 = StringToFloat(aippCfg.at(AIPP_VAR_RECI_CHN_3));
    }
}

float AippTiling::StringToFloat(string str)
{
    float f = 0;
    try {
        f = std::stof(str);
    } catch (const std::invalid_argument& e) {
        OP_LOGE(context_->GetNodeName(), "StringToFloat failed, invalid_argument, input str is: %s.", str.c_str());
        return f;
    } catch (const std::out_of_range& e) {
        OP_LOGE(context_->GetNodeName(), "StringToFloat failed, out_of_range, input str is: %s.", str.c_str());
        return f;
    }
    return f;
}

void AippTiling::SetSySTilingData()
{
    int64_t totalNum = tilingData.batchNum * tilingData.channelNum * tilingData.inputSizeH * tilingData.inputSizeW;
    int64_t numBlocks = CeilDiv(totalNum, MAX_THREAD_NUM);
    auto compileInfo = reinterpret_cast<const AippCompileInfo*>(context_->GetCompileInfo());
    int32_t coreNums = compileInfo->coreNum;
    if (numBlocks > coreNums) {
        context_->SetBlockDim(coreNums);
    } else {
        context_->SetBlockDim(numBlocks);
    }
    int64_t simtLocalMem = compileInfo->ubSize - DCACHE_SIZE;
    context_->SetLocalMemorySize(simtLocalMem);
}

ge::graphStatus AippTiling::CheckTilingData()
{
    OP_CHECK_IF(
        (tilingData.batchNum <= 0), OP_LOGE(context_->GetNodeName(), "Params of batchNum should not smaller than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.channelNum <= 0),
        OP_LOGE(context_->GetNodeName(), "Params of channelNum should not smaller than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.inputSizeW <= 0),
        OP_LOGE(context_->GetNodeName(), "Params of inputSizeW should not smaller than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.inputSizeH <= 0),
        OP_LOGE(context_->GetNodeName(), "Params of inputSizeH should not smaller than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.inputFormat <= 0),
        OP_LOGE(context_->GetNodeName(), "Params of inputFormat should not smaller than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.imageFormat <= 0),
        OP_LOGE(context_->GetNodeName(), "Params of imageFormat should not smaller than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.cropSwitch != 0 && tilingData.cropSwitch != 1),
        OP_LOGE(context_->GetNodeName(), "Params of cropSwitch should be 0/1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.cropStartPosH > tilingData.inputSizeH),
        OP_LOGE(context_->GetNodeName(), "Params of cropStartPosH should smaller than inputSizeH."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.cropStartPosW > tilingData.inputSizeW),
        OP_LOGE(context_->GetNodeName(), "Params of cropStartPosW should smaller than inputSizeW."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((tilingData.cropSizeH + tilingData.cropStartPosH) > tilingData.inputSizeH),
        OP_LOGE(context_->GetNodeName(), "Params of cropStartPosH + cropSizeH should smaller than inputSizeH."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((tilingData.cropSizeW + tilingData.cropStartPosW) > tilingData.inputSizeW),
        OP_LOGE(context_->GetNodeName(), "Params of cropStartPosW + cropSizeW should smaller than inputSizeW."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.cscSwitch != 0 && tilingData.cscSwitch != 1),
        OP_LOGE(context_->GetNodeName(), "Params of cscSwitch should be 0/1."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void AippTiling::PrintTilingData()
{
    stringstream ss;
    ss << "inputFormat: " << static_cast<uint32_t>(tilingData.inputFormat)
       << ", imageFormat: " << static_cast<uint32_t>(tilingData.imageFormat)
       << ", batchNum: " << tilingData.batchNum << ", channelNum: " << tilingData.channelNum
       << ", inputSizeW: " << tilingData.inputSizeW << ", inputSizeH: " << tilingData.inputSizeH
       << ", cropSwitch: " << static_cast<uint32_t>(tilingData.cropSwitch)
       << ", cropStartPosH: " << tilingData.cropStartPosH << ", cropStartPosW: " << tilingData.cropStartPosW
       << ", cropSizeH: " << tilingData.cropSizeH << ", cropSizeW: " << tilingData.cropSizeW
       << ", cscSwitch: " << static_cast<int32_t>(tilingData.cscSwitch)
       << ", cscMatrix00: " << static_cast<int32_t>(tilingData.cscMatrix00)
       << ", cscMatrix01: " << static_cast<int32_t>(tilingData.cscMatrix01)
       << ", cscMatrix02: " << static_cast<int32_t>(tilingData.cscMatrix02)
       << ", cscMatrix10: " << static_cast<int32_t>(tilingData.cscMatrix10)
       << ", cscMatrix11: " << static_cast<int32_t>(tilingData.cscMatrix11)
       << ", cscMatrix12: " << static_cast<int32_t>(tilingData.cscMatrix12)
       << ", cscMatrix20: " << static_cast<int32_t>(tilingData.cscMatrix20)
       << ", cscMatrix21: " << static_cast<int32_t>(tilingData.cscMatrix21)
       << ", cscMatrix22: " << static_cast<int32_t>(tilingData.cscMatrix22);
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());

    stringstream strs;
    strs << "outBias0: " << static_cast<int32_t>(tilingData.outBias0)
         << ", outBias1: " << static_cast<int32_t>(tilingData.outBias1)
         << ", outBias2: " << static_cast<int32_t>(tilingData.outBias2)
         << ", inBias0: " << static_cast<int32_t>(tilingData.inBias0)
         << ", inBias1: " << static_cast<int32_t>(tilingData.inBias1)
         << ", inBias2: " << static_cast<int32_t>(tilingData.inBias2)
         << ", dtcPixelMeanChn0: " << static_cast<int32_t>(tilingData.dtcPixelMeanChn0)
         << ", dtcPixelMeanChn1: " << static_cast<int32_t>(tilingData.dtcPixelMeanChn1)
         << ", dtcPixelMeanChn2: " << static_cast<int32_t>(tilingData.dtcPixelMeanChn2)
         << ", dtcPixelMeanChn3: " << static_cast<int32_t>(tilingData.dtcPixelMeanChn3)
         << ", dtcPixelMinChn0: " << tilingData.dtcPixelMinChn0 << ", dtcPixelMinChn1: " << tilingData.dtcPixelMinChn1
         << ", dtcPixelMinChn2: " << tilingData.dtcPixelMinChn2 << ", dtcPixelMinChn3: " << tilingData.dtcPixelMinChn3
         << ", dtcPixelVarReciChn0: " << tilingData.dtcPixelVarReciChn0
         << ", dtcPixelVarReciChn1: " << tilingData.dtcPixelVarReciChn1
         << ", dtcPixelVarReciChn2: " << tilingData.dtcPixelVarReciChn2
         << ", dtcPixelVarReciChn3: " << tilingData.dtcPixelVarReciChn3;
    OP_LOGI(context_->GetNodeName(), "%s", strs.str().c_str());
}

ge::graphStatus AippTiling::PostTiling()
{
    auto compileInfo = reinterpret_cast<const AippCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    SetSySTilingData();

    void* tilingBuf = context_->GetRawTilingData()->GetData();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    if (tilingBuf == nullptr || tilingBufCap < sizeof(tilingData)) {
        OP_LOGE(
            context_->GetNodeName(), "tiling buffer null or capacity too small, cap=%zu need=%zu.", tilingBufCap,
            sizeof(tilingData));
        return ge::GRAPH_FAILED;
    }
    errno_t cpyRet = memcpy_s(tilingBuf, tilingBufCap, &tilingData, sizeof(tilingData));
    if (cpyRet != 0) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s tiling data failed, ret=%d.", cpyRet);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForAipp(gert::TilingContext* context)
{
    return Ops::Cv::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForAipp(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForAipp running.");

    auto compileInfo = context->GetCompiledInfo<AippCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForAipp end.");
    return ge::GRAPH_SUCCESS;
}
REGISTER_TILING_TEMPLATE("Aipp", AippTiling, 1000);

IMPL_OP_OPTILING(Aipp).Tiling(TilingForAipp).TilingParse<AippCompileInfo>(TilingPrepareForAipp);
} // namespace optiling