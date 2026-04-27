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
#include "log/log.h"
#include "aipp_tiling.h"

namespace optiling {
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

inline static bool IsJsonSymbol(char c)
{
    const std::string symbols = "\"{}[](),:;";
    return symbols.find(c) != std::string::npos;
}

template <typename T>
inline static bool StringToNum(const std::string& str, T& number)
{
    std::istringstream iss(str);
    if (iss >> number) {
        return true;
    }
    return false;
}

template <typename T>
ge::graphStatus AippTiling::ParseNumAndValidateRange(
    const std::map<std::string, T*>& valueMap, int64_t minValue, int64_t maxValue)
{
    const std::string parseErrorLog = " failed to parse.";
    const std::string rangeErrorLog =
        " should between [" + std::to_string(minValue) + ", " + std::to_string(maxValue) + "].";

    for (const auto& item : valueMap) {
        if (aippCfg.find(item.first) == aippCfg.end()) {
            continue;
        }

        OP_CHECK_IF(!StringToNum(aippCfg.at(item.first), *item.second),
            OP_LOGE(context_->GetNodeName(), "%s", (item.first + parseErrorLog).c_str()), return ge::GRAPH_FAILED);

        OP_CHECK_IF(*item.second > maxValue || *item.second < minValue,
            OP_LOGE(context_->GetNodeName(), "%s", (item.first + rangeErrorLog).c_str()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AippTiling::GetTilingKey() const
{
    const bool cscSwitch = static_cast<bool>(tilingData.cscParam.cscSwitch);
    const bool isRgbFormat = (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_RGB888_U8) ||
                              tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_XRGB8888_U8));
    const bool isYuvFormat = (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8) ||
                              tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV400_U8));

    if (isGray) {
        if (isRgbFormat) {
            return AIPP_RGB_TO_GRAY;
        }
        if (isYuvFormat) {
            return AIPP_YUV_TO_GRAY;
        }
    }

    if (isRgbFormat) {
        return cscSwitch ? AIPP_RGB_TO_YUV : AIPP_RGB_PASS_THROUGH;
    }

    if (isYuvFormat) {
        return cscSwitch ? AIPP_YUV_TO_RGB : AIPP_YUV_PASS_THROUGH;
    }

    OP_LOGE(context_->GetNodeName(), "tilingKey is:0, please check aipp.cfg's inputformat and csc_switch");
    return AIPP_ERROR_TILINGKEY;
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

    OP_CHECK_IF(CheckAippCfg() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "aipp cfg is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetImagesValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetImagesValue fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputImage() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputImage fail."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetCropValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetCropValue fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCropSize() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckCropSize fail."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetCscValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetCscValue fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetDTCValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetDTCValue fail."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetPaddingValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetPaddingValue fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckPaddingSize() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckPaddingSize fail."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputFormat()
{
    auto inputImges = context_->GetInputDesc(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputImges);
    auto inputFormatLocal = inputImges->GetOriginFormat();
    OP_CHECK_IF(inputFormatLocal != ge::FORMAT_NHWC, OP_LOGE(context_->GetNodeName(),\
        "aipp input format only support NHWC."), return ge::GRAPH_FAILED);

    auto outputImages = context_->GetOutputDesc(OUTPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputImages);
    auto outputFormatLocal = outputImages->GetOriginFormat();
    OP_CHECK_IF(outputFormatLocal != ge::FORMAT_NCHW && outputFormatLocal != ge::FORMAT_NHWC,
        OP_LOGE(context_->GetNodeName(), "aipp output format only support NCHW and NHWC."), return ge::GRAPH_FAILED);

    auto inputShapePtr = context_->GetInputShape(INPUT_IMAGES_IDX);
    
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShapePtr);
    auto inputStorageShape = inputShapePtr->GetStorageShape();
    tilingData.batchNum = inputStorageShape.GetDim(IMAGE_BATCH_DIM);
    tilingData.channelNum = inputStorageShape.GetDim(NHWC_IMAGE_CHANNEL_DIM);
    tilingData.inputSizeH = inputStorageShape.GetDim(NHWC_IMAGE_H_DIM);
    tilingData.inputSizeW = inputStorageShape.GetDim(NHWC_IMAGE_W_DIM);
    tilingData.outputFormat = (outputFormatLocal == ge::FORMAT_NCHW) ? NCHW_FORMAT_INDEX : NHWC_FORMAT_INDEX;
    inputImageSize = inputStorageShape.GetShapeSize();

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputStorageShape = outputShapePtr->GetStorageShape();
    tilingData.outputSizeH = tilingData.outputFormat == NCHW_FORMAT_INDEX ?
                        outputStorageShape.GetDim(NCHW_IMAGE_H_DIM) :
                        outputStorageShape.GetDim(NHWC_IMAGE_H_DIM);
    tilingData.outputSizeW = tilingData.outputFormat == NCHW_FORMAT_INDEX ?
                        outputStorageShape.GetDim(NCHW_IMAGE_W_DIM) :
                        outputStorageShape.GetDim(NHWC_IMAGE_W_DIM);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckAippCfg()
{
    OP_CHECK_IF(aippCfg.size() == 0, 
        OP_LOGE(context_->GetNodeName(), "aippCfg is empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(aippCfg.find(AIPP_MODE) == aippCfg.end(),
        OP_LOGE(context_->GetNodeName(), "aippCfg has no aipp_mode."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(aippCfg.at(AIPP_MODE) != AIPP_MODE_STATIC,
        OP_LOGE(context_->GetNodeName(), "aippCfg value is not static."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::SetImagesValue()
{
    if (aippCfg.find(AIPP_INPUT_FORMAT) != aippCfg.end()) {
        OP_CHECK_IF(
            IMAGE_FORMAT_MAP.find(aippCfg.at(AIPP_INPUT_FORMAT)) == IMAGE_FORMAT_MAP.end(),
            OP_LOGE(context_->GetNodeName(), "aipp input_format only support RGB888_U8 or XRGB8888_U8 or YUV420SP_U8 or YUV400_U8."),
            return ge::GRAPH_FAILED);

        tilingData.imageFormat = IMAGE_FORMAT_MAP.at(aippCfg.at(AIPP_INPUT_FORMAT));
    } else {
        OP_LOGE(context_->GetNodeName(), "can not find input_format in aipp config.");
        return ge::GRAPH_FAILED;
    }

    if (aippCfg.find(AIPP_SRC_IMAGE_SIZE_H) != aippCfg.end()) {
        OP_CHECK_IF(!StringToNum(aippCfg.at(AIPP_SRC_IMAGE_SIZE_H), tilingData.inputSizeH),
            OP_LOGE(context_->GetNodeName(), "aipp src_image_size_h failed to parse."), return ge::GRAPH_FAILED);
    }

    if (aippCfg.find(AIPP_SRC_IMAGE_SIZE_W) != aippCfg.end()) {
        OP_CHECK_IF(!StringToNum(aippCfg.at(AIPP_SRC_IMAGE_SIZE_W), tilingData.inputSizeW),
            OP_LOGE(context_->GetNodeName(), "aipp src_image_size_w failed to parse."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputImage()
{
    OP_CHECK_IF(tilingData.inputSizeH == 1 || tilingData.inputSizeH > MAX_IMAGE_HIGH,
        OP_LOGE(context_->GetNodeName(), "The value range of the src_image_size_h is 0, [2, 4096]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData.inputSizeW > MAX_IMAGE_WIDTH,
        OP_LOGE(context_->GetNodeName(), "The value range of the src_image_size_w is [0, 4096]."),
        return ge::GRAPH_FAILED);

    stringstream errorCheckLog;
    errorCheckLog << "when input_format is " << aippCfg.at(AIPP_INPUT_FORMAT)
                  << ", input image size should be bigger than N * src_image_size_w * src_image_size_h * ";
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8)) {
        errorCheckLog << IMAGE_FORMAT_YUV420SP_U8_SIZE_LIMIT;
        OP_CHECK_IF((inputImageSize * CONST_VALUE_TWO <
            tilingData.batchNum * tilingData.inputSizeW * tilingData.inputSizeH * CONST_VALUE_THREE),
            OP_LOGE(context_->GetNodeName(), "%s", errorCheckLog.str().c_str()), return ge::GRAPH_FAILED);

        OP_CHECK_IF(tilingData.inputSizeW % EVEN_NUMBER_BASE != 0 || tilingData.inputSizeH % EVEN_NUMBER_BASE != 0,
            OP_LOGE(context_->GetNodeName(), \
            "When input_format is YUV420SP_U8, src_image_size_h/w must be even number"), return ge::GRAPH_FAILED);
    } else if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV400_U8)) {
        errorCheckLog << static_cast<int>(IMAGE_FORMAT_YUV400_U8_SIZE_LIMIT);
        OP_CHECK_IF((inputImageSize <
             tilingData.batchNum * tilingData.inputSizeW * tilingData.inputSizeH * IMAGE_FORMAT_YUV400_U8_SIZE_LIMIT),
            OP_LOGE(context_->GetNodeName(), "%s", errorCheckLog.str().c_str()), return ge::GRAPH_FAILED);
    } else if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_RGB888_U8)) {
        errorCheckLog << static_cast<int>(IMAGE_FORMAT_RGB888_U8_SIZE_LIMIT);
        OP_CHECK_IF((inputImageSize <
             tilingData.batchNum * tilingData.inputSizeW * tilingData.inputSizeH * IMAGE_FORMAT_RGB888_U8_SIZE_LIMIT),
            OP_LOGE(context_->GetNodeName(), "%s", errorCheckLog.str().c_str()), return ge::GRAPH_FAILED);
    } else {
        errorCheckLog << static_cast<int>(IMAGE_FORMAT_XRGB8888_U8_SIZE_LIMIT);
        OP_CHECK_IF((inputImageSize <
             tilingData.batchNum * tilingData.inputSizeW * tilingData.inputSizeH * IMAGE_FORMAT_XRGB8888_U8_SIZE_LIMIT),
            OP_LOGE(context_->GetNodeName(), "%s", errorCheckLog.str().c_str()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckInputDtype()
{
    auto inputImges = context_->GetInputDesc(INPUT_IMAGES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputImges);
    OP_CHECK_IF(inputImges->GetDataType() != ge::DT_UINT8, \
        OP_LOGE(context_->GetNodeName(), "inputImges only support uint8."), return ge::GRAPH_FAILED);

    auto outputImages = context_->GetOutputDesc(OUTPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputImages);
    OP_CHECK_IF(outputImages->GetDataType() != ge::DT_FLOAT16 && outputImages->GetDataType() != ge::DT_UINT8,
        OP_LOGE(context_->GetNodeName(), "outputImages only support float16 or uint8."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::ValidPaddingValue(float padValue, ge::DataType outputDtype)
{
    if (outputDtype == ge::DT_UINT8) {
        OP_CHECK_IF(tilingData.paddingParam.padValue < MIN_PADDING_VALUE_UINT8 ||
            tilingData.paddingParam.padValue > MAX_PADDING_VALUE_UINT8,
            OP_LOGE(context_->GetNodeName(), "padding value should between [0, 255] for uint8 output."),
            return ge::GRAPH_FAILED);
    } else if (outputDtype == ge::DT_FLOAT16) {
        OP_CHECK_IF(tilingData.paddingParam.padValue < MIN_PADDING_VALUE_FP16 ||
            tilingData.paddingParam.padValue > MAX_PADDING_VALUE_FP16,
            OP_LOGE(context_->GetNodeName(), "padding value should between [-65504, 65504] for fp16 output."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::SetPaddingValue()
{
    if (aippCfg.find(AIPP_PADDING) == aippCfg.end()) {
        tilingData.paddingParam.paddingSwitch = 0;
        return ge::GRAPH_SUCCESS;
    }

    if (aippCfg.at(AIPP_PADDING) != "true" && aippCfg.at(AIPP_PADDING) != "false") {
        return ge::GRAPH_FAILED;
    }

    if (aippCfg.at(AIPP_PADDING) == "false") {
        tilingData.paddingParam.paddingSwitch = 0;
        return ge::GRAPH_SUCCESS;
    }

    tilingData.paddingParam.paddingSwitch = 1;

    const std::map<string, int32_t*> PADDING_SIZE_MAP = {
        {AIPP_LEFT_PADDING_SIZE, &tilingData.paddingParam.leftPaddingSize},
        {AIPP_RIGHT_PADDING_SIZE, &tilingData.paddingParam.rightPaddingSize},
        {AIPP_TOP_PADDING_SIZE, &tilingData.paddingParam.topPaddingSize},
        {AIPP_BOTTOM_PADDING_SIZE, &tilingData.paddingParam.bottomPaddingSize}
    };

    OP_CHECK_IF(ParseNumAndValidateRange(PADDING_SIZE_MAP, 0, MAX_PADDING_SIZE) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "PADDING_SIZE_MAP parse and validate range failed."),
        return ge::GRAPH_FAILED);

    if (aippCfg.find(AIPP_PADDING_VALUE) != aippCfg.end()) {
        OP_CHECK_IF(!StringToNum(aippCfg.at(AIPP_PADDING_VALUE), tilingData.paddingParam.padValue),
            OP_LOGE(context_->GetNodeName(), "AIPP_PADDING_VALUE failed to parse."),
            return ge::GRAPH_FAILED);
        
        auto outputImages = context_->GetOutputDesc(OUTPUT_FEATURES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputImages);
        OP_CHECK_IF(ValidPaddingValue(tilingData.paddingParam.padValue, outputImages->GetDataType()) !=
            ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "PadValue is invalidate."),
            return ge::GRAPH_FAILED);
    } else {
        tilingData.paddingParam.padValue = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckPaddingSize()
{
    if (!static_cast<bool>(tilingData.paddingParam.paddingSwitch)) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t expectedH = tilingData.cropParam.cropSizeH +
                        tilingData.paddingParam.topPaddingSize +
                        tilingData.paddingParam.bottomPaddingSize;
    uint32_t expectedW = tilingData.cropParam.cropSizeW +
                        tilingData.paddingParam.leftPaddingSize +
                        tilingData.paddingParam.rightPaddingSize;
    OP_CHECK_IF(expectedH != tilingData.outputSizeH || expectedW != tilingData.outputSizeW,
        OP_LOGE(context_->GetNodeName(),
        "output size mismatch: output=%dx%d,expectedOut=crop(%dx%d)+padding(%d,%d,%d,%d)=%dx%d.",
        tilingData.outputSizeH, tilingData.outputSizeW,
        tilingData.cropParam.cropSizeH, tilingData.cropParam.cropSizeW,
        tilingData.paddingParam.topPaddingSize, tilingData.paddingParam.bottomPaddingSize,
        tilingData.paddingParam.leftPaddingSize, tilingData.paddingParam.rightPaddingSize,
        expectedH, expectedW),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData.outputSizeW > MAX_PADDING_OUT_W,
        OP_LOGE(context_->GetNodeName(),
        "After padding, aipp output w[%d] should be less than or eaqual to 1080.",
        tilingData.outputSizeW),
        return ge::GRAPH_FAILED);
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
            key.erase(std::remove_if(key.begin(), key.end(), IsJsonSymbol), key.end());
            string value = line.substr(colonPos + 1);
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            value.erase(std::remove_if(value.begin(), value.end(), IsJsonSymbol), value.end());
            aippConfig[key] = value;
        }
    }
    return aippConfig;
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
    SetGrayFlag();
    SwapChannelForCSC();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AippTiling::SwapChannelForCSC()
{
    if ((tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_RGB888_U8) ||
        tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_XRGB8888_U8)) &&
        tilingData.cscParam.rbuvSwapSwitch == 1) {
        swap(tilingData.cscParam.cscMatrix00, tilingData.cscParam.cscMatrix02);
        swap(tilingData.cscParam.cscMatrix10, tilingData.cscParam.cscMatrix12);
        swap(tilingData.cscParam.cscMatrix20, tilingData.cscParam.cscMatrix22);
    }
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8) &&
        tilingData.cscParam.rbuvSwapSwitch == 1) {
        swap(tilingData.cscParam.cscMatrix01, tilingData.cscParam.cscMatrix02);
        swap(tilingData.cscParam.cscMatrix11, tilingData.cscParam.cscMatrix12);
        swap(tilingData.cscParam.cscMatrix21, tilingData.cscParam.cscMatrix22);
    }
}

void AippTiling::SetGrayFlag()
{
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV400_U8) &&
        !static_cast<bool>(tilingData.cscParam.cscSwitch)) {
        isGray = true;
        return;
    }

    bool anyMatrix1NotZero = (tilingData.cscParam.cscMatrix10 != 0) || 
                             (tilingData.cscParam.cscMatrix11 != 0) || 
                             (tilingData.cscParam.cscMatrix12 != 0);
    bool anyMatrix2NotZero = (tilingData.cscParam.cscMatrix20 != 0) || 
                             (tilingData.cscParam.cscMatrix21 != 0) || 
                             (tilingData.cscParam.cscMatrix22 != 0);
    if (anyMatrix1NotZero || anyMatrix2NotZero) {
        return;
    }
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_RGB888_U8) ||
        tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_XRGB8888_U8)) {
        if ((tilingData.cscParam.outBias0 == 0) &&
            (tilingData.cscParam.outBias1 == 0) && 
            (tilingData.cscParam.outBias2 == 0)) {
            isGray = true;
        }
    }
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8) &&
        tilingData.cscParam.cscMatrix01 == 0 && tilingData.cscParam.cscMatrix02 == 0) {
        if ((tilingData.cscParam.inBias0 == 0) && 
            (tilingData.cscParam.inBias1 == 0) && 
            (tilingData.cscParam.inBias2 == 0)) {
            isGray = true;
        }
    }
}

ge::graphStatus AippTiling::SetCropValue()
{
    if ((aippCfg.find(AIPP_CROP) != aippCfg.end()) && (aippCfg.at(AIPP_CROP) == "true")) {
        tilingData.cropParam.cropSwitch = 1;
        const std::map<string, uint32_t*> CROP_START_MAP = {
            {AIPP_CROP_LOAD_START_POS_W, &tilingData.cropParam.cropStartPosW},
            {AIPP_CROP_LOAD_START_POS_H, &tilingData.cropParam.cropStartPosH}
        };

        const std::map<string, uint32_t*> CROP_SIZE_MAP = {
            {AIPP_CROP_SIZE_W, &tilingData.cropParam.cropSizeW},
            {AIPP_CROP_SIZE_H, &tilingData.cropParam.cropSizeH}
        };

        OP_CHECK_IF(
            ParseNumAndValidateRange(CROP_START_MAP, 0, MAX_IMAGE_HIGH - 1) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CROP_START_MAP parse and validate range failed."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            ParseNumAndValidateRange(CROP_SIZE_MAP, 0, MAX_IMAGE_HIGH) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CROP_SIZE_MAP parse and validate range failed."),
            return ge::GRAPH_FAILED);
    } else {
        tilingData.cropParam.cropSwitch = 0;
        tilingData.cropParam.cropSizeH = tilingData.inputSizeH;
        tilingData.cropParam.cropSizeW = tilingData.inputSizeW;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::CheckCropSize()
{
    if (static_cast<bool>(tilingData.cropParam.cropSwitch)) {
        OP_CHECK_IF(tilingData.cropParam.cropSizeW > tilingData.inputSizeW,
            OP_LOGE(context_->GetNodeName(), "aipp.cfg's crop_size_w should smaller than src_image_size_w."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(tilingData.cropParam.cropSizeH > tilingData.inputSizeH,
            OP_LOGE(context_->GetNodeName(), "aipp.cfg's crop_size_h should smaller than src_image_size_h."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(tilingData.cropParam.cropStartPosW + tilingData.cropParam.cropSizeW > tilingData.inputSizeW,
            OP_LOGE(context_->GetNodeName(), \
            "aipp.cfg's crop_start_pos_w + crop_size_w should smaller than src_image_size_w."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            tilingData.cropParam.cropStartPosH + tilingData.cropParam.cropSizeH > tilingData.inputSizeH,
            OP_LOGE(context_->GetNodeName(), \
                "aipp.cfg's crop_start_pos_h + crop_size_h should smaller than src_image_size_h."),
            return ge::GRAPH_FAILED);

        if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8)) {
            OP_CHECK_IF( tilingData.cropParam.cropStartPosW % EVEN_NUMBER_BASE != 0 ||
                tilingData.cropParam.cropStartPosH % EVEN_NUMBER_BASE != 0, \
                OP_LOGE( context_->GetNodeName(), \
                "When input_format is YUV420SP_U8, crop_start_pos_w/h must be even number"),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}


void AippTiling::SetCscDefaultMatrix()
{
    tilingData.cscParam.cscMatrix00 = CSC_IDENTITY_SCALE; 
    tilingData.cscParam.cscMatrix01 = 0; 
    tilingData.cscParam.cscMatrix02 = 0;
    tilingData.cscParam.cscMatrix10 = 0; 
    tilingData.cscParam.cscMatrix11 = CSC_IDENTITY_SCALE; 
    tilingData.cscParam.cscMatrix12 = 0;
    tilingData.cscParam.cscMatrix20 = 0; 
    tilingData.cscParam.cscMatrix21 = 0; 
    tilingData.cscParam.cscMatrix22 = CSC_IDENTITY_SCALE;
    tilingData.cscParam.inBias0 = 0;  
    tilingData.cscParam.inBias1 = 0;  
    tilingData.cscParam.inBias2 = 0;
    tilingData.cscParam.outBias0 = 0; 
    tilingData.cscParam.outBias1 = 0; 
    tilingData.cscParam.outBias2 = 0;
}

void AippTiling::SetCscFormatBias()
{
    if (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV420SP_U8)) {
        tilingData.cscParam.outBias0 = 0;
        tilingData.cscParam.outBias1 = 0;
        tilingData.cscParam.outBias2 = 0;
    } else {
        tilingData.cscParam.inBias0 = 0;
        tilingData.cscParam.inBias1 = 0;
        tilingData.cscParam.inBias2 = 0;
    }
}

ge::graphStatus AippTiling::SetCscValue()
{
    if ((aippCfg.find(AIPP_CFG_CSC_SWITCH) != aippCfg.end()) && (aippCfg.at(AIPP_CFG_CSC_SWITCH) == "true")) {
        tilingData.cscParam.cscSwitch = 1;

        OP_CHECK_IF(
            tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_YUV400_U8), OP_LOGE(context_->GetNodeName(), 
            "When input format is YUV400_U8, it doesn't make sense to convert to RGB, csc_switch : true."),
            return ge::GRAPH_FAILED);

        const std::map<string, int16_t*> CSC_MATRIX_MAP = {
            {AIPP_CFG_MATRIX_R0C0, &tilingData.cscParam.cscMatrix00},
            {AIPP_CFG_MATRIX_R0C1, &tilingData.cscParam.cscMatrix01},
            {AIPP_CFG_MATRIX_R0C2, &tilingData.cscParam.cscMatrix02},
            {AIPP_CFG_MATRIX_R1C0, &tilingData.cscParam.cscMatrix10},
            {AIPP_CFG_MATRIX_R1C1, &tilingData.cscParam.cscMatrix11},
            {AIPP_CFG_MATRIX_R1C2, &tilingData.cscParam.cscMatrix12},
            {AIPP_CFG_MATRIX_R2C0, &tilingData.cscParam.cscMatrix20},
            {AIPP_CFG_MATRIX_R2C1, &tilingData.cscParam.cscMatrix21},
            {AIPP_CFG_MATRIX_R2C2, &tilingData.cscParam.cscMatrix22},
        };
        const std::map<string, int16_t*> CSC_BIAS_MAP = {
            {AIPP_CFG_OUTPUT_BIAS_0, &tilingData.cscParam.outBias0},
            {AIPP_CFG_OUTPUT_BIAS_1, &tilingData.cscParam.outBias1},
            {AIPP_CFG_OUTPUT_BIAS_2, &tilingData.cscParam.outBias2},
            {AIPP_CFG_INPUT_BIAS_0, &tilingData.cscParam.inBias0},
            {AIPP_CFG_INPUT_BIAS_1, &tilingData.cscParam.inBias1},
            {AIPP_CFG_INPUT_BIAS_2, &tilingData.cscParam.inBias2},
        };
        OP_CHECK_IF(
            ParseNumAndValidateRange(CSC_MATRIX_MAP, MIN_MATRIX_BOUND, MAX_MATRIX_BOUND) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CSC_MATRIX_MAP parse and validate range failed."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(ParseNumAndValidateRange(CSC_BIAS_MAP, 0, MAX_RGB_BOUND) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CSC_BIAS_MAP parse and validate range failed."),
            return ge::GRAPH_FAILED);
    } else {
        SetCscDefaultMatrix();
    }

    SetCscFormatBias();

    return SetSwapSwitch();
}

ge::graphStatus AippTiling::SetSwapSwitch()
{
    if (aippCfg.find(AIPP_RBUV_SWAP_SWITCH) != aippCfg.end() && aippCfg.at(AIPP_RBUV_SWAP_SWITCH) == "true") {
        tilingData.cscParam.rbuvSwapSwitch = 1;
    }

    if (aippCfg.find(AIPP_AX_SWAP_SWITCH) != aippCfg.end() && aippCfg.at(AIPP_AX_SWAP_SWITCH) == "true") {
        if (tilingData.imageFormat != IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_XRGB8888_U8)) {
            OP_LOGE(context_->GetNodeName(), "Only XRGB888_U8 supports ax_swap_switch being true.");
            return ge::GRAPH_FAILED;
        }
        tilingData.cscParam.axSwapSwitch = 1;
    }

    tilingData.srcChannelOffset =
        (tilingData.imageFormat == IMAGE_FORMAT_MAP.at(IMAGE_FORMAT_XRGB8888_U8) &&
         tilingData.cscParam.axSwapSwitch == 1) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AippTiling::SetDTCValue()
{
    const std::map<string, int16_t*> DTC_MEAN_MAP = {
        {AIPP_MEAN_CHN_0, &tilingData.dtcParam.dtcPixelMeanChn0},
        {AIPP_MEAN_CHN_1, &tilingData.dtcParam.dtcPixelMeanChn1},
        {AIPP_MEAN_CHN_2, &tilingData.dtcParam.dtcPixelMeanChn2},
        {AIPP_MEAN_CHN_3, &tilingData.dtcParam.dtcPixelMeanChn3},
    };

    const std::map<string, float*> DTC_MIN_RECI_MAP = {
        {AIPP_MIN_CHN_0, &tilingData.dtcParam.dtcPixelMinChn0},
        {AIPP_MIN_CHN_1, &tilingData.dtcParam.dtcPixelMinChn1},
        {AIPP_MIN_CHN_2, &tilingData.dtcParam.dtcPixelMinChn2},
        {AIPP_MIN_CHN_3, &tilingData.dtcParam.dtcPixelMinChn3},
        {AIPP_VAR_RECI_CHN_0, &tilingData.dtcParam.dtcPixelVarReciChn0},
        {AIPP_VAR_RECI_CHN_1, &tilingData.dtcParam.dtcPixelVarReciChn1},
        {AIPP_VAR_RECI_CHN_2, &tilingData.dtcParam.dtcPixelVarReciChn2},
        {AIPP_VAR_RECI_CHN_3, &tilingData.dtcParam.dtcPixelVarReciChn3},
    };

    OP_CHECK_IF(
        ParseNumAndValidateRange(DTC_MEAN_MAP, 0, MAX_RGB_BOUND) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DTC_MEAN_MAP parse and validate range failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ParseNumAndValidateRange(DTC_MIN_RECI_MAP, MIN_CHN_BOUND, MAX_CHN_BOUND) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DTC_MIN_RECI_MAP parse and validate range failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
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

void AippTiling::PrintTilingData() const
{
    stringstream ss;
    ss << "imageFormat: " << static_cast<int>(tilingData.imageFormat)
       << ", outputFormat: " << static_cast<int>(tilingData.outputFormat)
       << ", batchNum: " << tilingData.batchNum << ", channelNum: " << tilingData.channelNum
       << ", rbuvSwapSwitch: " << tilingData.cscParam.rbuvSwapSwitch
       << ", axSwapSwitch: " << tilingData.cscParam.axSwapSwitch
       << ", inputSizeW: " << tilingData.inputSizeW << ", inputSizeH: " << tilingData.inputSizeH
       << ", cropSwitch: " << tilingData.cropParam.cropSwitch
       << ", cropStartPosH: " << tilingData.cropParam.cropStartPosH
       << ", cropStartPosW: " << tilingData.cropParam.cropStartPosW
       << ", cropSizeH: " << tilingData.cropParam.cropSizeH
       << ", cropSizeW: " << tilingData.cropParam.cropSizeW << ", cscSwitch: " << tilingData.cscParam.cscSwitch
       << ", cscMatrix00: " << tilingData.cscParam.cscMatrix00 << ", cscMatrix01: " << tilingData.cscParam.cscMatrix01
       << ", cscMatrix02: " << tilingData.cscParam.cscMatrix02 << ", cscMatrix10: " << tilingData.cscParam.cscMatrix10
       << ", cscMatrix11: " << tilingData.cscParam.cscMatrix11 << ", cscMatrix12: " << tilingData.cscParam.cscMatrix12
       << ", cscMatrix20: " << tilingData.cscParam.cscMatrix20 << ", cscMatrix21: " << tilingData.cscParam.cscMatrix21
       << ", cscMatrix22: " << tilingData.cscParam.cscMatrix22
       << ", paddingSwitch: " << tilingData.paddingParam.paddingSwitch;
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());

    stringstream strs;
    strs << "outBias0: " << tilingData.cscParam.outBias0 << ", outBias1: " << tilingData.cscParam.outBias1
         << ", outBias2: " << tilingData.cscParam.outBias2 << ", inBias0: " << tilingData.cscParam.inBias0
         << ", inBias1: " << tilingData.cscParam.inBias1 << ", inBias2: " << tilingData.cscParam.inBias2
         << ", dtcPixelMeanChn0: " << tilingData.dtcParam.dtcPixelMeanChn0
         << ", dtcPixelMeanChn1: " << tilingData.dtcParam.dtcPixelMeanChn1
         << ", dtcPixelMeanChn2: " << tilingData.dtcParam.dtcPixelMeanChn2
         << ", dtcPixelMeanChn3: " << tilingData.dtcParam.dtcPixelMeanChn3
         << ", dtcPixelMinChn0: " << tilingData.dtcParam.dtcPixelMinChn0
         << ", dtcPixelMinChn1: " << tilingData.dtcParam.dtcPixelMinChn1
         << ", dtcPixelMinChn2: " << tilingData.dtcParam.dtcPixelMinChn2
         << ", dtcPixelMinChn3: " << tilingData.dtcParam.dtcPixelMinChn3
         << ", dtcPixelVarReciChn0: " << tilingData.dtcParam.dtcPixelVarReciChn0
         << ", dtcPixelVarReciChn1: " << tilingData.dtcParam.dtcPixelVarReciChn1
         << ", dtcPixelVarReciChn2: " << tilingData.dtcParam.dtcPixelVarReciChn2
         << ", dtcPixelVarReciChn3: " << tilingData.dtcParam.dtcPixelVarReciChn3;
    OP_LOGI(context_->GetNodeName(), "%s", strs.str().c_str());

    stringstream strp;
    strp << "leftPaddingSize: " << tilingData.paddingParam.leftPaddingSize
        << ", rightPaddingSize: " << tilingData.paddingParam.rightPaddingSize
        << ", topPaddingSize: " << tilingData.paddingParam.topPaddingSize
        << ", bottomPaddingSize: " << tilingData.paddingParam.bottomPaddingSize
        << ", padValue: " << tilingData.paddingParam.padValue;
    OP_LOGI(context_->GetNodeName(), "%s", strp.str().c_str());
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