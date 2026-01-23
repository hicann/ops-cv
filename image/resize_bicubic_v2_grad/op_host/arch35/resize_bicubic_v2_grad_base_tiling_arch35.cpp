/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_bicubic_v2_grad_base_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_base_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"
#include "tiling_base/tiling_util.h"
#include "util/platform_util.h"

namespace optiling {

constexpr size_t NUM_0 = 0;
constexpr size_t NUM_1 = 1;
constexpr size_t NUM_2 = 2;
constexpr size_t NUM_3 = 3;
constexpr size_t NUM_4 = 4;
constexpr int64_t RSV_BLOCK_NUM = 8;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;

bool ResizeBicubicV2GradBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo_.coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo_.ubSize = ubSize;
    } else {
        auto compileInfo = reinterpret_cast<const ResizeBicubicV2GradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        compileInfo_.coreNum = compileInfo->coreNum;
        compileInfo_.ubSize = compileInfo->ubSize;
    }

    compileInfo_.ubBlockSize = Ops::Base::GetUbBlockSize(context_);

    OP_CHECK_IF(
        compileInfo_.coreNum <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get core num"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compileInfo_.ubSize <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get ub size"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compileInfo_.ubBlockSize <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get ub block size"),
        return ge::GRAPH_FAILED);

    compileInfo_.isDetermine = context_->GetDeterministic() == 1 ? 1 : 0;

    calcInfo_.ubBlockNum = Ops::Base::CeilDiv(compileInfo_.ubSize, compileInfo_.ubBlockSize) - RSV_BLOCK_NUM;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetTensorInfo()
{
    auto gradsShapePtr = context_->GetInputShape(NUM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsShapePtr);
    inputInfo_.gradsShape = Ops::Cv::OpTiling::EnsureNotScalar(gradsShapePtr->GetOriginShape());
    auto gradsDescPtr = context_->GetInputDesc(NUM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsDescPtr);
    inputInfo_.gradsDtype = gradsDescPtr->GetDataType();
    inputInfo_.gradsFormat = gradsDescPtr->GetOriginFormat();

    auto originalImageShapePtr = context_->GetInputShape(NUM_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, originalImageShapePtr);
    inputInfo_.originalImageShape = Ops::Cv::OpTiling::EnsureNotScalar(originalImageShapePtr->GetOriginShape());
    auto originalImageDescPtr = context_->GetInputDesc(NUM_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, originalImageDescPtr);
    inputInfo_.originalImageDtype = originalImageDescPtr->GetDataType();
    inputInfo_.originalImageFormat = originalImageDescPtr->GetOriginFormat();

    auto yShapePtr = context_->GetOutputShape(NUM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    inputInfo_.yShape = Ops::Cv::OpTiling::EnsureNotScalar(yShapePtr->GetOriginShape());
    auto yDescPtr = context_->GetOutputDesc(NUM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDescPtr);
    inputInfo_.yDtype = yDescPtr->GetDataType();
    inputInfo_.yFormat = yDescPtr->GetOriginFormat();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckDtypeValid()
{
    OP_CHECK_IF(
        inputInfo_.gradsDtype != ge::DT_FLOAT && inputInfo_.gradsDtype != ge::DT_FLOAT16 &&
            inputInfo_.gradsDtype != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "Data type of grads must be FLOAT or FLOAT16 or BFLOAT16."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputInfo_.originalImageDtype != inputInfo_.gradsDtype || inputInfo_.yDtype != inputInfo_.gradsDtype,
        OP_LOGE(context_->GetNodeName(), "Data types of grads, original_image and y must be same."),
        return ge::GRAPH_FAILED);

    calcInfo_.gradsDtypeSize = GetSizeByDataType(inputInfo_.gradsDtype);
    calcInfo_.yDtypeSize = GetSizeByDataType(inputInfo_.yDtype);
    OP_CHECK_IF(
        calcInfo_.gradsDtypeSize <= 0 || calcInfo_.yDtypeSize <= 0,
        OP_LOGE(context_->GetNodeName(), "grads or y dtype size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckFormatValid()
{
    OP_CHECK_IF(
        inputInfo_.gradsFormat != ge::FORMAT_NCHW && inputInfo_.gradsFormat != ge::FORMAT_NHWC && inputInfo_.gradsFormat != ge::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "grads format must be NCHW or NHWC or ND"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputInfo_.originalImageFormat != inputInfo_.gradsFormat ||
            inputInfo_.originalImageFormat != inputInfo_.yFormat,
        OP_LOGE(context_->GetNodeName(), "Formats of grads, original_image and y must be same."),
        return ge::GRAPH_FAILED);

    inputInfo_.format = inputInfo_.yFormat == ge::FORMAT_NHWC ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckShapeValid()
{
    OP_CHECK_IF(
        inputInfo_.gradsShape.GetDimNum() != NUM_4 || inputInfo_.yShape.GetDimNum() != NUM_4,
        OP_LOGE(context_->GetNodeName(), "Shapes of grads and y must be 4D."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputInfo_.originalImageShape != inputInfo_.yShape,
        OP_LOGE(context_->GetNodeName(), "Shapes of original_image and y must be same."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputInfo_.gradsShape.GetDim(NUM_0) != inputInfo_.yShape.GetDim(NUM_0),
        OP_LOGE(context_->GetNodeName(), "N dims of grads and y must be same."), return ge::GRAPH_FAILED);
    inputInfo_.lenN = inputInfo_.gradsShape.GetDim(NUM_0);

    if (inputInfo_.gradsFormat == ge::FORMAT_NCHW || inputInfo_.gradsFormat == ge::FORMAT_ND) {
        OP_CHECK_IF(
            inputInfo_.gradsShape.GetDim(NUM_1) != inputInfo_.yShape.GetDim(NUM_1),
            OP_LOGE(context_->GetNodeName(), "C dims of grads and y must be same."), return ge::GRAPH_FAILED);
        inputInfo_.lenC = inputInfo_.gradsShape.GetDim(NUM_1);
        inputInfo_.lenSrcH = inputInfo_.yShape.GetDim(NUM_2);
        inputInfo_.lenSrcW = inputInfo_.yShape.GetDim(NUM_3);
        inputInfo_.lenDstH = inputInfo_.gradsShape.GetDim(NUM_2);
        inputInfo_.lenDstW = inputInfo_.gradsShape.GetDim(NUM_3);
    } else if (inputInfo_.gradsFormat == ge::FORMAT_NHWC) {
        OP_CHECK_IF(
            inputInfo_.gradsShape.GetDim(NUM_3) != inputInfo_.yShape.GetDim(NUM_3),
            OP_LOGE(context_->GetNodeName(), "C dims of grads and y must be same."), return ge::GRAPH_FAILED);
        inputInfo_.lenSrcH = inputInfo_.yShape.GetDim(NUM_1);
        inputInfo_.lenSrcW = inputInfo_.yShape.GetDim(NUM_2);
        inputInfo_.lenDstH = inputInfo_.gradsShape.GetDim(NUM_1);
        inputInfo_.lenDstW = inputInfo_.gradsShape.GetDim(NUM_2);
        inputInfo_.lenC = inputInfo_.gradsShape.GetDim(NUM_3);
    }

    OP_CHECK_IF(
        inputInfo_.lenN <= 0 || inputInfo_.lenC <= 0,
        OP_LOGE(context_->GetNodeName(), "N and C dims of grads and y must be greater than 0."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputInfo_.lenSrcH <= 0 || inputInfo_.lenSrcW <= 0 || inputInfo_.lenDstH <= 0 || inputInfo_.lenDstW <= 0,
        OP_LOGE(context_->GetNodeName(), "H and W dims of grads and y must be greater than 0."),
        return ge::GRAPH_FAILED);

    calcInfo_.gradsShapeSize = inputInfo_.gradsShape.GetShapeSize();
    calcInfo_.yShapeSize = inputInfo_.yShape.GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t attrNum = attrs->GetAttrNum();
    if (attrNum > NUM_0) {
        inputInfo_.alignCorners = *(attrs->GetAttrPointer<bool>(NUM_0)) ? 1 : 0;
    }
    if (attrNum > NUM_1) {
        auto scales = attrs->GetAttrPointer<gert::ContinuousVector>(NUM_1);
        OP_CHECK_IF(
            scales->GetSize() != NUM_2,
            OP_LOGE(context_->GetNodeName(), "Scales size %ld is invalid.", scales->GetSize()),
            return ge::GRAPH_FAILED);
        const float* scalesData = reinterpret_cast<const float*>(scales->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, scalesData);
        inputInfo_.oriScaleH = scalesData[NUM_0];
        inputInfo_.oriScaleW = scalesData[NUM_1];
        OP_LOGI(context_->GetNodeName(), "original scales(%f, %f)", inputInfo_.oriScaleH, inputInfo_.oriScaleW);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        (GetTensorInfo() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetTensorInfo failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckDtypeValid() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckDtypeValid failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckFormatValid() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckFormatValid failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckShapeValid() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckShapeValid failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (GetAttrInfo() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetAttrInfo failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeBicubicV2GradBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

void ResizeBicubicV2GradBaseTiling::SetScales()
{
    if (inputInfo_.alignCorners) {
        if (calcInfo_.isMatchDetermine) {
            if (inputInfo_.lenSrcH > 1 && inputInfo_.lenDstH > 1) {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH - 1) / (inputInfo_.lenDstH - 1);
                calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH - 1) / (inputInfo_.lenSrcH - 1);
            } else {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH) / inputInfo_.lenDstH;
                calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH) / (inputInfo_.lenSrcH);
            }
            if (inputInfo_.lenSrcW > 1 && inputInfo_.lenDstW > 1) {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW - 1) / (inputInfo_.lenDstW - 1);
                calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW - 1) / (inputInfo_.lenSrcW - 1);
            } else {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW) / inputInfo_.lenDstW;
                calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW) / (inputInfo_.lenSrcW);
            }
        } else {
            if (inputInfo_.lenDstH > 1) {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH - 1) / (inputInfo_.lenDstH - 1);
            } else {
                calcInfo_.scaleH = 0.0f;
            }
            if (inputInfo_.lenDstW > 1) {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW - 1) / (inputInfo_.lenDstW - 1);
            } else {
                calcInfo_.scaleW = 0.0f;
            }
        }
    } else {
        if (inputInfo_.oriScaleH > 0.0f) {
            calcInfo_.scaleH = 1.0f / inputInfo_.oriScaleH;
            calcInfo_.inverseScaleH = inputInfo_.oriScaleH;
        } else {
            calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH) / inputInfo_.lenDstH;
            calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH) / inputInfo_.lenSrcH;
        }
        if (inputInfo_.oriScaleW > 0.0f) {
            calcInfo_.scaleW = 1.0f / inputInfo_.oriScaleW;
            calcInfo_.inverseScaleW = inputInfo_.oriScaleW;
        } else {
            calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW) / inputInfo_.lenDstW;
            calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW) / inputInfo_.lenSrcW;
        }
    }
}

bool ResizeBicubicV2GradBaseTiling::IsUseIdx32() const
{
    return calcInfo_.gradsShapeSize <= UINT32_MAX && calcInfo_.yShapeSize <= UINT32_MAX &&
           inputInfo_.lenSrcH <= INT32_MAX && inputInfo_.lenSrcW <= INT32_MAX;
}

ge::graphStatus Tiling4ResizeBicubicV2Grad(gert::TilingContext* context)
{
    return Ops::Cv::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4ResizeBicubicV2Grad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4ResizeBicubicV2Grad start");

    auto compileInfo = context->GetCompiledInfo<ResizeBicubicV2GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;

    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4ResizeBicubicV2Grad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ResizeBicubicV2Grad)
    .Tiling(Tiling4ResizeBicubicV2Grad)
    .TilingParse<ResizeBicubicV2GradCompileInfo>(TilingPrepare4ResizeBicubicV2Grad);
} // namespace optiling
