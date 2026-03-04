/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max_tiling.cpp
 * \brief Ascend950(arch35) tiling implementation
 */

#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "roi_pooling_with_arg_max_tiling_arch35.h"
#include "../../op_kernel/arch35/roi_pooling_with_arg_max_tiling_key.h"
#include "util/math_util.h"

namespace optiling {

constexpr uint32_t NCHW_DIMS = 4;
constexpr uint32_t ROI_DIMS = 2;
constexpr uint32_t ROI_DIM_2_SHAPE = 5;
constexpr uint32_t NHWC_H_DIM = 2;
constexpr uint32_t NHWC_W_DIM = 3;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ATTR_2 = 2;
constexpr uint64_t ATTR_3 = 3;
constexpr uint64_t TYPE_MODE1 = 0;  // fp32
constexpr uint64_t TYPE_MODE2 = 1;  // fp16
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MAX_THREAD_NUM = 1024;

void RoiPoolingWithArgMaxTiling::SetTilingData()
{
    int64_t outputDataCount = roiNumber_ * channels_ * poolH_ * poolW_;
    useCoreNum_ = static_cast<int64_t>(
        Ops::Base::CeilDiv(outputDataCount, Ops::Base::CeilDiv(outputDataCount, totalCoreNum_)));
    useCoreNum_ = (useCoreNum_ <= 0) ? 1 : useCoreNum_;
    context_->SetBlockDim(static_cast<uint32_t>(useCoreNum_));

    tilingData_->channels = channels_;
    tilingData_->fmHeight = fmHeight_;
    tilingData_->fmWidth = fmWidth_;
    tilingData_->roiNumber = roiNumber_;
    tilingData_->poolH = poolH_;
    tilingData_->poolW = poolW_;
    tilingData_->spatialH = spatialH_;
    tilingData_->spatialW = spatialW_;
}

ge::graphStatus RoiPoolingWithArgMaxTiling::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const auto *poolHPtr = attrs->GetAttrPointer<int64_t>(ATTR_0);
    const auto *poolWPtr = attrs->GetAttrPointer<int64_t>(ATTR_1);
    const auto *spatialHPtr = attrs->GetAttrPointer<float>(ATTR_2);
    const auto *spatialWPtr = attrs->GetAttrPointer<float>(ATTR_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolHPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolWPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, spatialHPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, spatialWPtr);

    poolH_ = *poolHPtr;
    poolW_ = *poolWPtr;
    spatialH_ = *spatialHPtr;
    spatialW_ = *spatialWPtr;

    OP_CHECK_IF(
        poolH_ <= 0 || poolW_ <= 0,
        OP_LOGE(context_, "pooled_h, pooled_w must be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        spatialH_ <= 0 || spatialW_ <= 0,
        OP_LOGE(context_, "spatial_scale_h, spatial_scale_w must be greater than 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingWithArgMaxTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        totalCoreNum_ <= 0,
        OP_LOGE(context_, "Failed to get core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= 0UL,
        OP_LOGE(context_, "ubSize must be greater than zero, but is %lu", ubSize),
        return ge::GRAPH_FAILED);
    auto localMemorySize = context_->SetLocalMemorySize(ubSize - DCACHE_SIZE);
    OP_LOGD(context_, "ubSize = %lu, localMemorySize = %d.", ubSize, localMemorySize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingWithArgMaxTiling::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<RoiPoolingWithArgMaxRegBaseTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr,
            OP_LOGE(context_, "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(
             tilingData_, sizeof(RoiPoolingWithArgMaxRegBaseTilingData), 0,
             sizeof(RoiPoolingWithArgMaxRegBaseTilingData)) != EOK),
        OP_LOGE(context_, "memset tilingdata failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool RoiPoolingWithArgMaxTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
            dataTypeTilingKey_ = TYPE_MODE1;
            break;
        case ge::DT_FLOAT16:
            dataTypeTilingKey_ = TYPE_MODE2;
            break;
        default:
            return false;
    }
    return true;
}

void RoiPoolingWithArgMaxTiling::SetTilingKey()
{
    OP_LOGI(context_, "dtype is %lu", dataTypeTilingKey_);
    tilingKey_ = GET_TPL_TILING_KEY(dataTypeTilingKey_);
}

uint64_t RoiPoolingWithArgMaxTiling::GetTilingKey()
{
    return tilingKey_;
}

ge::graphStatus RoiPoolingWithArgMaxTiling::GetInputTensorInfo()
{
    constexpr uint32_t INPUT_FM_IDX = 0;
    constexpr uint32_t INPUT_ROIS_IDX = 1;

    auto inputFM = context_->GetInputShape(INPUT_FM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputFM);
    gert::Shape inputFMShape = Ops::Cv::OpTiling::EnsureNotScalar(inputFM->GetStorageShape());
    OP_CHECK_IF(
        inputFMShape.GetDimNum() != NCHW_DIMS,
        OP_LOGE(context_, "input0 shape dim = %zu, should be 4.", inputFMShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    channels_ = static_cast<int64_t>(inputFMShape.GetDim(1));
    fmHeight_ = static_cast<int64_t>(inputFMShape.GetDim(NHWC_H_DIM));
    fmWidth_ = static_cast<int64_t>(inputFMShape.GetDim(NHWC_W_DIM));

    auto inputRoi = context_->GetInputShape(INPUT_ROIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputRoi);
    gert::Shape inputRoiShape = Ops::Cv::OpTiling::EnsureNotScalar(inputRoi->GetStorageShape());
    OP_CHECK_IF(
        inputRoiShape.GetDimNum() != ROI_DIMS,
        OP_LOGE(context_, "input1 ROI shape dim = %zu, should be 2.", inputRoiShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputRoiShape.GetDim(1) != ROI_DIM_2_SHAPE,
        OP_LOGE(context_, "input1 ROI dim1 = %zu, should be 5.", inputRoiShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    roiNumber_ = static_cast<int64_t>(inputRoiShape.GetDim(0));

    auto inputFMDesc = context_->GetInputDesc(INPUT_FM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputFMDesc);
    inputDType_ = inputFMDesc->GetDataType();
    OP_CHECK_IF(
        GetDataTypeKey(inputDType_) == false,
        OP_LOGE(context_, "The dtype of input x must be in [float32, float16]."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingWithArgMaxTiling::DoTiling()
{
    OP_LOGD(context_, "Enter RoiPoolingWithArgMaxTiling DoTiling");

    OP_CHECK_IF(
        SetAttrParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "SetAttrParams failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetInputTensorInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "GetInputTensorInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "GetPlatformInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InitTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "InitTilingData failed."),
        return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context_->SetTilingKey(GetTilingKey());

    OP_LOGI(context_, "tiling data:%s.", tilingData_->toString().c_str());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGD(context_, "End dotiling");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForRoiPoolingWithArgMax(gert::TilingContext *context)
{
    OP_LOGD(context, "TilingForRoiPoolingWithArgMax running begin.");

    RoiPoolingWithArgMaxTiling tilingObj(context);

    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepareForRoiPoolingWithArgMax([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RoiPoolingWithArgMax)
    .Tiling(TilingForRoiPoolingWithArgMax)
    .TilingParse<RoiPoolingWithArgMaxCompileInfo>(TilingPrepareForRoiPoolingWithArgMax);

}  // namespace optiling
