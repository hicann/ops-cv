/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aipp_infershape.cpp
 * \brief RT2 InferShape / InferShapeRange for Aipp.
 *        Does not rewrite images, SetSize, or SetAttr.
 *        images rewrite at compile: historical SOC = RT1 AippInfer;
 *        Ascend950 = kAfterInferShape pass after this Infer (not this file).
 */

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/range.h"
#include "graph/types.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t kImagesIdx = 0;
constexpr size_t kParamsIdx = 1;
constexpr size_t kFeaturesIdx = 0;
constexpr size_t kAttrCfgIdx = 0;

constexpr uint8_t kChannel1 = 1;
constexpr uint8_t kChannel3 = 3;
constexpr uint8_t kChannel4 = 4;

constexpr uint8_t kFormatYuv420sp = 1;
constexpr uint8_t kFormatXrgb8888 = 2;
constexpr uint8_t kFormatRgb888 = 5;
constexpr uint8_t kFormatArgb8888 = 6;
constexpr uint8_t kFormatAyuv444 = 9;
constexpr uint8_t kFormatYuv400 = 10;

constexpr int32_t kParamsHeadSize = 64;
constexpr int32_t kBatchNumIndex = 4;
constexpr int32_t kSrcImageSizeWIndex = 8;
constexpr int32_t kSrcImageSizeHIndex = 12;
constexpr int32_t kScfSwitchIndex = 1;
constexpr int32_t kPaddingSwitchIndex = 2;
constexpr int32_t kCropStartPosWIndex = 8;
constexpr int32_t kCropStartPosHIndex = 12;
constexpr int32_t kCropSizeWIndex = 16;
constexpr int32_t kCropSizeHIndex = 20;
constexpr int32_t kScfInputSizeWIndex = 24;
constexpr int32_t kScfInputSizeHIndex = 28;
constexpr int32_t kScfOutputSizeWIndex = 32;
constexpr int32_t kScfOutputSizeHIndex = 36;
constexpr int32_t kPaddingSizeTopIndex = 40;
constexpr int32_t kPaddingSizeBottomIndex = 44;
constexpr int32_t kPaddingSizeLeftIndex = 48;
constexpr int32_t kPaddingSizeRightIndex = 52;
constexpr int32_t kParamsMinBytes = kParamsHeadSize + kPaddingSizeRightIndex + static_cast<int32_t>(sizeof(int32_t));

constexpr int64_t kMinCropSize = 8;
constexpr int64_t kNum4 = 4;
constexpr int64_t kNum5 = 5;

constexpr size_t kBatchDim = 0;
constexpr size_t kNchwDimC = 1;
constexpr size_t kNchwDimH = 2;
constexpr size_t kNchwDimW = 3;
constexpr size_t kNhwcDimH = 1;
constexpr size_t kNhwcDimW = 2;
constexpr size_t kNhwcDimC = 3;
constexpr size_t kC04DimC1 = 1;
constexpr size_t kC04DimH = 2;
constexpr size_t kC04DimW = 3;
constexpr size_t kC04DimC0 = 4;

constexpr const char* kAttrAippMode = "aipp_mode";
constexpr const char* kAttrMaxSrcImgSize = "max_src_image_size";
constexpr const char* kAttrCrop = "crop";
constexpr const char* kAttrCropSizeH = "crop_size_h";
constexpr const char* kAttrCropSizeW = "crop_size_w";
constexpr const char* kAttrResize = "resize";
constexpr const char* kAttrResizeOutputH = "resize_output_h";
constexpr const char* kAttrResizeOutputW = "resize_output_w";
constexpr const char* kAttrSrcImageSizeH = "src_image_size_h";
constexpr const char* kAttrSrcImageSizeW = "src_image_size_w";
constexpr const char* kAttrPadding = "padding";
constexpr const char* kAttrLeftPaddingSize = "left_padding_size";
constexpr const char* kAttrRightPaddingSize = "right_padding_size";
constexpr const char* kAttrTopPaddingSize = "top_padding_size";
constexpr const char* kAttrBottomPaddingSize = "bottom_padding_size";
constexpr const char* kAttrInputFormat = "input_format";
constexpr const char* kValueAippModeDynamic = "dynamic";
constexpr const char* kValueAippModeStatic = "static";

struct AippParams {
    uint8_t inputFormat = 0;
    int8_t batchNum = 0;
    int32_t srcImageSizeW = 0;
    int32_t srcImageSizeH = 0;
    int8_t cropSwitch = 0;
    int8_t scfSwitch = 0;
    int8_t paddingSwitch = 0;
    int32_t cropStartPosW = 0;
    int32_t cropStartPosH = 0;
    int32_t cropSizeW = 0;
    int32_t cropSizeH = 0;
    int32_t scfInputSizeW = 0;
    int32_t scfInputSizeH = 0;
    int32_t scfOutputSizeW = 0;
    int32_t scfOutputSizeH = 0;
    int32_t paddingSizeTop = 0;
    int32_t paddingSizeBottom = 0;
    int32_t paddingSizeLeft = 0;
    int32_t paddingSizeRight = 0;
};

struct AippParamsCfg {
    std::string aippMode;
    std::string inputFormat;
    int64_t maxSrcImageSize = 0;
    bool crop = false;
    int64_t cropSizeH = 0;
    int64_t cropSizeW = 0;
    bool resize = false;
    int64_t resizeOutputH = 0;
    int64_t resizeOutputW = 0;
    int64_t srcImageSizeH = 0;
    int64_t srcImageSizeW = 0;
    bool padding = false;
    int64_t leftPaddingSize = 0;
    int64_t rightPaddingSize = 0;
    int64_t topPaddingSize = 0;
    int64_t bottomPaddingSize = 0;
};

struct ImageNhwc {
    int64_t n = 0;
    int64_t c = 0;
    int64_t h = 0;
    int64_t w = 0;
};

struct StaticFeaturesIn {
    ge::Format imagesFormat = ge::FORMAT_RESERVED;
    const gert::Shape* imagesShape = nullptr;
    gert::Shape* featuresShape = nullptr;
};

struct ShapeRangeIn {
    const gert::Range<gert::Shape>* imagesRange = nullptr;
    ge::Format imagesFormat = ge::FORMAT_RESERVED;
    gert::Range<gert::Shape>* featuresRange = nullptr;
};

inline ge::Format ToPrimaryFormat(ge::Format format) { return static_cast<ge::Format>(ge::GetPrimaryFormat(format)); }

inline bool IsUnknownRankShape(const gert::Shape& shape) { return Ops::Base::IsUnknownRank(shape); }

template <typename T>
T ReadParam(const uint8_t* data, int32_t byteOffset)
{
    return *reinterpret_cast<const T*>(data + byteOffset);
}

void UseConfiguredHw(bool enabled, int64_t configuredH, int64_t configuredW, int64_t& height, int64_t& width)
{
    if (!enabled) {
        return;
    }
    height = configuredH ? configuredH : height;
    width = configuredW ? configuredW : width;
}

bool IsHwAtLeastMin(const char* nodeName, const char* heightName, const char* widthName, int64_t height, int64_t width)
{
    if (height >= kMinCropSize && width >= kMinCropSize) {
        return true;
    }
    const std::string paramNames = std::string(heightName) + ", " + widthName;
    const std::string values = "[" + std::to_string(height) + ", " + std::to_string(width) + "]";
    const std::string reason = "must be greater than or equal to " + std::to_string(kMinCropSize);
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, paramNames.c_str(), values.c_str(), reason.c_str());
    return false;
}

ge::graphStatus CheckMaxSrcImageSize(const char* nodeName, int64_t maxSrcImageSize)
{
    if (maxSrcImageSize > 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "max_src_image_size", std::to_string(maxSrcImageSize).c_str(),
                                          "max_src_image_size must be greater than 0");
    return ge::GRAPH_FAILED;
}

template <class T>
ge::graphStatus GetAippAttrValue(const char* nodeName, const nlohmann::json& cfgJson, const char* attrName, T& value)
{
    if (cfgJson.find(attrName) == cfgJson.end()) {
        OP_LOGD(nodeName, "Attr: %s is not in json object", attrName);
        return ge::GRAPH_FAILED;
    }
    value = cfgJson[attrName];
    return ge::GRAPH_SUCCESS;
}

// RT1 AippInfer: json::parse, then is_object. Parse throws into AippInfer's catch.
ge::graphStatus ParseAippConfigObject(const char* nodeName, const char* configData, nlohmann::json& cfgJson)
{
    cfgJson = nlohmann::json::parse(configData);
    OP_CHECK_IF(!cfgJson.is_object(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aipp_config_path", configData,
                                                      "aipp_config_path must be a json object"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <class Context>
ge::graphStatus LoadAippConfigJson(Context* context, nlohmann::json& cfgJson, const char** configData)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* data = attrs->template GetAttrPointer<char>(kAttrCfgIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    if (configData != nullptr) {
        *configData = data;
    }
    return ParseAippConfigObject(context->GetNodeName(), data, cfgJson);
}

ge::graphStatus FillParamsCfg(const char* nodeName, const nlohmann::json& cfgJson, AippParamsCfg& params)
{
    if (GetAippAttrValue(nodeName, cfgJson, kAttrAippMode, params.aippMode) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "aipp_mode");
        return ge::GRAPH_FAILED;
    }
    if (params.aippMode == kValueAippModeDynamic) {
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrMaxSrcImgSize, params.maxSrcImageSize);
        return ge::GRAPH_SUCCESS;
    }
    if (params.aippMode != kValueAippModeStatic) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "aipp_mode", params.aippMode.c_str(), "static or dynamic");
        return ge::GRAPH_FAILED;
    }
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrInputFormat, params.inputFormat);
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrCrop, params.crop);
    if (params.crop) {
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrCropSizeH, params.cropSizeH);
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrCropSizeW, params.cropSizeW);
    }
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrResize, params.resize);
    if (params.resize) {
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrResizeOutputH, params.resizeOutputH);
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrResizeOutputW, params.resizeOutputW);
    }
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrSrcImageSizeH, params.srcImageSizeH);
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrSrcImageSizeW, params.srcImageSizeW);
    (void)GetAippAttrValue(nodeName, cfgJson, kAttrPadding, params.padding);
    if (params.padding) {
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrLeftPaddingSize, params.leftPaddingSize);
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrRightPaddingSize, params.rightPaddingSize);
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrTopPaddingSize, params.topPaddingSize);
        (void)GetAippAttrValue(nodeName, cfgJson, kAttrBottomPaddingSize, params.bottomPaddingSize);
    }
    return ge::GRAPH_SUCCESS;
}

void GetOutputHeightWidth(const AippParamsCfg& aippOpParams, int64_t& outputHeight, int64_t& outputWidth)
{
    UseConfiguredHw(aippOpParams.crop, aippOpParams.cropSizeH, aippOpParams.cropSizeW, outputHeight, outputWidth);
    UseConfiguredHw(aippOpParams.resize, aippOpParams.resizeOutputH, aippOpParams.resizeOutputW, outputHeight,
                    outputWidth);
    UseConfiguredHw(!aippOpParams.crop && !aippOpParams.resize, aippOpParams.srcImageSizeH, aippOpParams.srcImageSizeW,
                    outputHeight, outputWidth);
    if (aippOpParams.padding) {
        outputHeight = outputHeight + aippOpParams.topPaddingSize + aippOpParams.bottomPaddingSize;
        outputWidth = outputWidth + aippOpParams.leftPaddingSize + aippOpParams.rightPaddingSize;
    }
}

ge::graphStatus ParseImagesLayout(const char* nodeName, ge::Format imagesFormat, const gert::Shape& imagesShape,
                                  ImageNhwc& layout)
{
    const ge::Format fmt = ToPrimaryFormat(imagesFormat);
    const auto dimNum = static_cast<int64_t>(imagesShape.GetDimNum());
    if (((fmt == ge::FORMAT_NCHW || fmt == ge::FORMAT_NHWC) && dimNum < kNum4) ||
        (fmt == ge::FORMAT_NC1HWC0_C04 && dimNum < kNum5)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "images", std::to_string(dimNum).c_str(),
            "NCHW and NHWC need at least 4 dimensions, NC1HWC0_C04 needs at least 5 dimensions");
        return ge::GRAPH_FAILED;
    }
    if (fmt == ge::FORMAT_NCHW) {
        layout.n = imagesShape.GetDim(kBatchDim);
        layout.c = imagesShape.GetDim(kNchwDimC);
        layout.h = imagesShape.GetDim(kNchwDimH);
        layout.w = imagesShape.GetDim(kNchwDimW);
        return ge::GRAPH_SUCCESS;
    }
    if (fmt == ge::FORMAT_NHWC) {
        layout.n = imagesShape.GetDim(kBatchDim);
        layout.h = imagesShape.GetDim(kNhwcDimH);
        layout.w = imagesShape.GetDim(kNhwcDimW);
        layout.c = imagesShape.GetDim(kNhwcDimC);
        return ge::GRAPH_SUCCESS;
    }
    if (fmt == ge::FORMAT_NC1HWC0_C04) {
        layout.n = imagesShape.GetDim(kBatchDim);
        const int64_t c1 = imagesShape.GetDim(kC04DimC1);
        layout.h = imagesShape.GetDim(kC04DimH);
        layout.w = imagesShape.GetDim(kC04DimW);
        const int64_t c0 = imagesShape.GetDim(kC04DimC0);
        layout.c = (c1 == ge::UNKNOWN_DIM || c0 == ge::UNKNOWN_DIM) ? ge::UNKNOWN_DIM : (c1 * c0);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "images", Ops::Base::ToString(fmt).c_str(), "NCHW, NHWC or NC1HWC0_C04");
    return ge::GRAPH_FAILED;
}

void CopyShape(const gert::Shape& src, gert::Shape& dst)
{
    const size_t dimNum = src.GetDimNum();
    dst.SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        dst.SetDim(i, src.GetDim(i));
    }
}

ge::Format GetImagesStorageFormat(gert::InferShapeContext* context)
{
    const gert::Tensor* imagesTensor = context->GetInputTensor(kImagesIdx);
    if (imagesTensor != nullptr) {
        return ToPrimaryFormat(imagesTensor->GetStorageFormat());
    }
    const auto* imagesDesc = context->GetInputDesc(kImagesIdx);
    if (imagesDesc != nullptr) {
        return ToPrimaryFormat(imagesDesc->GetStorageFormat());
    }
    return ge::FORMAT_RESERVED;
}

ge::Format GetFeaturesStorageFormat(const gert::CompileTimeTensorDesc* desc)
{
    if (desc == nullptr) {
        return ge::FORMAT_RESERVED;
    }
    return ToPrimaryFormat(desc->GetStorageFormat());
}

int64_t GetDynamicShapeChannel(uint8_t inputFormat)
{
    switch (inputFormat) {
        case kFormatXrgb8888:
        case kFormatArgb8888:
        case kFormatAyuv444:
            return kChannel4;
        case kFormatYuv400:
            return kChannel1;
        default:
            return kChannel3;
    }
}

bool GetDynamicShapeOutputHW(const char* nodeName, const AippParams& aippParams, int64_t& outputH, int64_t& outputW)
{
    if (!IsHwAtLeastMin(nodeName, "srcImageSizeH", "srcImageSizeW", outputH, outputW)) {
        return false;
    }
    UseConfiguredHw(aippParams.cropSwitch > 0, aippParams.cropSizeH, aippParams.cropSizeW, outputH, outputW);
    if (aippParams.cropSwitch > 0 && !IsHwAtLeastMin(nodeName, "cropSizeH", "cropSizeW", outputH, outputW)) {
        return false;
    }
    UseConfiguredHw(aippParams.scfSwitch > 0, aippParams.scfOutputSizeH, aippParams.scfOutputSizeW, outputH, outputW);
    if (aippParams.paddingSwitch > 0) {
        outputH = outputH + aippParams.paddingSizeTop + aippParams.paddingSizeBottom;
        outputW = outputW + aippParams.paddingSizeLeft + aippParams.paddingSizeRight;
    }
    return true;
}

bool CheckImageInputFormat(const char* nodeName, const AippParams& aippParams)
{
    if (aippParams.inputFormat != kFormatYuv420sp && aippParams.inputFormat != kFormatYuv400 &&
        aippParams.inputFormat != kFormatRgb888 && aippParams.inputFormat != kFormatXrgb8888) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "inputFormat",
                                  std::to_string(static_cast<unsigned>(aippParams.inputFormat)).c_str(),
                                  "yuv420sp, yuv400, rgb888 or xrgb8888");
        return false;
    }
    return true;
}

void ParseAippParams(const uint8_t* constData, AippParams& aippParams)
{
    aippParams.inputFormat = *constData;
    aippParams.batchNum = ReadParam<int8_t>(constData, kBatchNumIndex);
    aippParams.srcImageSizeW = ReadParam<int32_t>(constData, kSrcImageSizeWIndex);
    aippParams.srcImageSizeH = ReadParam<int32_t>(constData, kSrcImageSizeHIndex);
    aippParams.cropSwitch = ReadParam<int8_t>(constData, kParamsHeadSize);
    aippParams.scfSwitch = ReadParam<int8_t>(constData, kParamsHeadSize + kScfSwitchIndex);
    aippParams.paddingSwitch = ReadParam<int8_t>(constData, kParamsHeadSize + kPaddingSwitchIndex);
    aippParams.cropStartPosW = ReadParam<int32_t>(constData, kParamsHeadSize + kCropStartPosWIndex);
    aippParams.cropStartPosH = ReadParam<int32_t>(constData, kParamsHeadSize + kCropStartPosHIndex);
    aippParams.cropSizeW = ReadParam<int32_t>(constData, kParamsHeadSize + kCropSizeWIndex);
    aippParams.cropSizeH = ReadParam<int32_t>(constData, kParamsHeadSize + kCropSizeHIndex);
    aippParams.scfInputSizeW = ReadParam<int32_t>(constData, kParamsHeadSize + kScfInputSizeWIndex);
    aippParams.scfInputSizeH = ReadParam<int32_t>(constData, kParamsHeadSize + kScfInputSizeHIndex);
    aippParams.scfOutputSizeW = ReadParam<int32_t>(constData, kParamsHeadSize + kScfOutputSizeWIndex);
    aippParams.scfOutputSizeH = ReadParam<int32_t>(constData, kParamsHeadSize + kScfOutputSizeHIndex);
    aippParams.paddingSizeTop = ReadParam<int32_t>(constData, kParamsHeadSize + kPaddingSizeTopIndex);
    aippParams.paddingSizeBottom = ReadParam<int32_t>(constData, kParamsHeadSize + kPaddingSizeBottomIndex);
    aippParams.paddingSizeLeft = ReadParam<int32_t>(constData, kParamsHeadSize + kPaddingSizeLeftIndex);
    aippParams.paddingSizeRight = ReadParam<int32_t>(constData, kParamsHeadSize + kPaddingSizeRightIndex);
}

ge::graphStatus InferStaticFeatures(const char* nodeName, const AippParamsCfg& params, const StaticFeaturesIn& in)
{
    ImageNhwc layout;
    if (ParseImagesLayout(nodeName, in.imagesFormat, *in.imagesShape, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t outputHeight = layout.h;
    int64_t outputWidth = layout.w;
    GetOutputHeightWidth(params, outputHeight, outputWidth);
    if (outputHeight != layout.h || outputWidth != layout.w) {
        const std::string shapes = "[" + std::to_string(layout.h) + "," + std::to_string(layout.w) + "], [" +
                                   std::to_string(outputHeight) + "," + std::to_string(outputWidth) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "features, aipp_output", shapes.c_str(),
                                               "features H and W must equal aipp output H and W");
        return ge::GRAPH_FAILED;
    }
    CopyShape(*in.imagesShape, *in.featuresShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDynamicModeFeatures(const char* nodeName, const AippParamsCfg& params,
                                         const gert::Shape& imagesShape, gert::Shape& featuresShape)
{
    if (CheckMaxSrcImageSize(nodeName, params.maxSrcImageSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsUnknownRankShape(imagesShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "images", "[-2]", "unknown rank is not supported");
        return ge::GRAPH_FAILED;
    }
    CopyShape(imagesShape, featuresShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDynamicRuntimeFeatures(const char* nodeName, const uint8_t* paramsData, ge::Format featuresFormat,
                                            gert::Shape& featuresShape)
{
    AippParams aippParams;
    ParseAippParams(paramsData, aippParams);
    if (!CheckImageInputFormat(nodeName, aippParams)) {
        return ge::GRAPH_FAILED;
    }
    const ge::Format fmt = ToPrimaryFormat(featuresFormat);
    if (fmt != ge::FORMAT_NCHW && fmt != ge::FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "features", Ops::Base::ToString(fmt).c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    const int64_t realChannel = GetDynamicShapeChannel(aippParams.inputFormat);
    int64_t outputH = aippParams.srcImageSizeH;
    int64_t outputW = aippParams.srcImageSizeW;
    if (!GetDynamicShapeOutputHW(nodeName, aippParams, outputH, outputW)) {
        return ge::GRAPH_FAILED;
    }
    featuresShape.SetDimNum(static_cast<size_t>(kNum4));
    if (fmt == ge::FORMAT_NCHW) {
        featuresShape.SetDim(kBatchDim, aippParams.batchNum);
        featuresShape.SetDim(kNchwDimC, realChannel);
        featuresShape.SetDim(kNchwDimH, outputH);
        featuresShape.SetDim(kNchwDimW, outputW);
        return ge::GRAPH_SUCCESS;
    }
    featuresShape.SetDim(kBatchDim, aippParams.batchNum);
    featuresShape.SetDim(kNhwcDimH, outputH);
    featuresShape.SetDim(kNhwcDimW, outputW);
    featuresShape.SetDim(kNhwcDimC, realChannel);
    return ge::GRAPH_SUCCESS;
}

bool IsDynamicRuntimeByAttr(const char* configData, const gert::Tensor* paramsTensor)
{
    if (configData == nullptr || paramsTensor == nullptr || paramsTensor->GetAddr() == nullptr) {
        return false;
    }
    const std::string cfgStr = configData;
    const int64_t modePosition = static_cast<int64_t>(cfgStr.find(kAttrAippMode));
    const int64_t dynamicPosition = static_cast<int64_t>(cfgStr.find(kValueAippModeDynamic));
    return modePosition > 0 && dynamicPosition > 0;
}

ge::graphStatus SetRangeDims(gert::Range<gert::Shape>* range, const gert::Shape& minShape, const gert::Shape& maxShape)
{
    OP_CHECK_IF(range == nullptr || range->GetMin() == nullptr || range->GetMax() == nullptr,
                OP_LOGE("Aipp", "%s is nullptr!", "features shape range"), return ge::GRAPH_FAILED);
    *range->GetMin() = minShape;
    *range->GetMax() = maxShape;
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForAipp(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Aipp", "%s is nullptr!", "InferShapeContext"), return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    try {
        OP_LOGI(nodeName, "AippInfer start");

        const char* configData = nullptr;
        nlohmann::json cfgJson;
        if (LoadAippConfigJson(context, cfgJson, &configData) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape* imagesShape = context->GetInputShape(kImagesIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, imagesShape);
        gert::Shape* featuresShape = context->GetOutputShape(kFeaturesIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
        const auto* featuresDesc = context->GetOutputDesc(kFeaturesIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
        const ge::Format imagesFormat = GetImagesStorageFormat(context);
        const ge::Format featuresFormat = GetFeaturesStorageFormat(featuresDesc);
        const gert::Tensor* paramsTensor = context->GetOptionalInputTensor(kParamsIdx);

        if (IsDynamicRuntimeByAttr(configData, paramsTensor)) {
            OP_LOGD(nodeName, "AippInfer, running, do DynamicShapeInfershape");
            const size_t paramsBytes = paramsTensor->GetSize();
            if (paramsBytes < static_cast<size_t>(kParamsMinBytes)) {
                const std::string reason = "params size must be at least " + std::to_string(kParamsMinBytes) + " bytes";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "params", std::to_string(paramsBytes).c_str(),
                                                      reason.c_str());
                return ge::GRAPH_FAILED;
            }
            const auto* paramsData = static_cast<const uint8_t*>(paramsTensor->GetAddr());
            return InferDynamicRuntimeFeatures(nodeName, paramsData, featuresFormat, *featuresShape);
        }
        AippParamsCfg params;
        if (FillParamsCfg(nodeName, cfgJson, params) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (params.aippMode == kValueAippModeDynamic) {
            return InferDynamicModeFeatures(nodeName, params, *imagesShape, *featuresShape);
        }
        const StaticFeaturesIn in{imagesFormat, imagesShape, featuresShape};
        return InferStaticFeatures(nodeName, params, in);
    } catch (...) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aipp_config_path", "invalid json",
                                              "failed to parse aipp_config_path");
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus InferDynamicShapeRange(const char* nodeName, const gert::Range<gert::Shape>* imagesRange,
                                              gert::Range<gert::Shape>* featuresRange)
{
    OP_CHECK_IF(imagesRange == nullptr || imagesRange->GetMin() == nullptr || imagesRange->GetMax() == nullptr,
                OP_LOGE(nodeName, "%s is nullptr!", "images shape range"), return ge::GRAPH_FAILED);
    return SetRangeDims(featuresRange, *imagesRange->GetMin(), *imagesRange->GetMax());
}

static ge::graphStatus InferStaticShapeRange(const char* nodeName, const AippParamsCfg& params, const ShapeRangeIn& in)
{
    OP_CHECK_IF(in.imagesRange == nullptr || in.imagesRange->GetMin() == nullptr || in.imagesRange->GetMax() == nullptr,
                OP_LOGE(nodeName, "%s is nullptr!", "images shape range"), return ge::GRAPH_FAILED);
    gert::Shape minShape;
    gert::Shape maxShape;
    const StaticFeaturesIn minIn{in.imagesFormat, in.imagesRange->GetMin(), &minShape};
    const StaticFeaturesIn maxIn{in.imagesFormat, in.imagesRange->GetMax(), &maxShape};
    if (InferStaticFeatures(nodeName, params, minIn) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (InferStaticFeatures(nodeName, params, maxIn) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SetRangeDims(in.featuresRange, minShape, maxShape);
}

static ge::Format GetImagesRangeFormat(gert::InferShapeRangeContext* context)
{
    const gert::Range<gert::Tensor>* imagesTensorRange = context->GetInputTensorRange(kImagesIdx);
    if (imagesTensorRange != nullptr && imagesTensorRange->GetMin() != nullptr) {
        return ToPrimaryFormat(imagesTensorRange->GetMin()->GetStorageFormat());
    }
    return GetFeaturesStorageFormat(context->GetInputDesc(kImagesIdx));
}

static const gert::Range<gert::Shape>* ResolveImagesShapeRange(gert::InferShapeRangeContext* context,
                                                               gert::Shape& minHolder, gert::Shape& maxHolder,
                                                               gert::Range<gert::Shape>& rangeHolder)
{
    const gert::Range<gert::Shape>* imagesRange = context->GetInputShapeRange(kImagesIdx);
    if (imagesRange != nullptr && imagesRange->GetMin() != nullptr && imagesRange->GetMax() != nullptr) {
        return imagesRange;
    }
    const gert::Range<gert::Tensor>* tensorRange = context->GetInputTensorRange(kImagesIdx);
    if (tensorRange == nullptr || tensorRange->GetMin() == nullptr || tensorRange->GetMax() == nullptr) {
        return nullptr;
    }
    minHolder = tensorRange->GetMin()->GetOriginShape();
    maxHolder = tensorRange->GetMax()->GetOriginShape();
    rangeHolder = gert::Range<gert::Shape>(&minHolder, &maxHolder);
    return &rangeHolder;
}

static ge::graphStatus InferShapeRangeForAipp(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Aipp", "%s is nullptr!", "InferShapeRangeContext"),
                return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    try {
        nlohmann::json cfgJson;
        if (LoadAippConfigJson(context, cfgJson, nullptr) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        AippParamsCfg params;
        if (FillParamsCfg(nodeName, cfgJson, params) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        gert::Shape imagesMinHolder;
        gert::Shape imagesMaxHolder;
        gert::Range<gert::Shape> imagesRangeHolder;
        const gert::Range<gert::Shape>* imagesRange = ResolveImagesShapeRange(context, imagesMinHolder, imagesMaxHolder,
                                                                              imagesRangeHolder);
        gert::Range<gert::Shape>* featuresRange = context->GetOutputShapeRange(kFeaturesIdx);
        OP_CHECK_IF(featuresRange == nullptr, OP_LOGE(nodeName, "%s is nullptr!", "features shape range"),
                    return ge::GRAPH_FAILED);

        const ge::Format imagesFormat = GetImagesRangeFormat(context);

        if (params.aippMode == kValueAippModeDynamic) {
            if (CheckMaxSrcImageSize(nodeName, params.maxSrcImageSize) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            return InferDynamicShapeRange(nodeName, imagesRange, featuresRange);
        }
        const ShapeRangeIn rangeIn{imagesRange, imagesFormat, featuresRange};
        return InferStaticShapeRange(nodeName, params, rangeIn);
    } catch (...) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aipp_config_path", "invalid json",
                                              "failed to parse aipp_config_path");
        return ge::GRAPH_FAILED;
    }
}

IMPL_OP_INFERSHAPE(Aipp)
    .InferShape(InferShapeForAipp)
    .InferShapeRange(InferShapeRangeForAipp)
    .InputsDataDependency({kParamsIdx});
} // namespace ops
