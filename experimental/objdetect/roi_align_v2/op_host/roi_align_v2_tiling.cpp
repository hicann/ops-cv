/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 # Authors (accounts):
 # - Liu Jun <@kbryantttt>
 # - Tu Yuanhang <@TuYHAAAAAA>
 # - Zhou Jianhua<@LePenseur>
 # - Liang Yanglin <@liang-yanglin>
 # - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file roi_align_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_data.h"
#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_key.h"

namespace optiling {

using namespace Ops::Cv::OpTiling;

struct RoiAlignV2CompileInfo {};

// tiling 分发入口
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiAlignV2TilingFunc(gert::TilingContext* context)
{   
    int64_t coreNum;
    uint64_t ubSize;
    
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    RoiAlignV2TilingData* tiling = context->GetTilingData<RoiAlignV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RoiAlignV2TilingData), 0, sizeof(RoiAlignV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    auto features_shape = context->GetInputShape(0)->GetStorageShape();
    auto rois_shape = context->GetInputShape(1)->GetStorageShape();

    uint32_t batch = features_shape.GetDim(0);
    uint32_t channels = features_shape.GetDim(1);
    uint32_t height = features_shape.GetDim(2);
    uint32_t width = features_shape.GetDim(3);
    uint32_t numRois = rois_shape.GetDim(0);
    uint32_t roiLength = rois_shape.GetDim(1);  
    
    auto attrs = context->GetAttrs();
    int32_t pooledHeight = 0;
    int32_t pooledWidth = 0;
    float spatialScale = 0.0f;
    int32_t samplingRatio = 0;  

    if (attrs != nullptr) {
        const int64_t* pooledHeightAttr = attrs->GetInt(0);
        if (pooledHeightAttr != nullptr) {
            pooledHeight = static_cast<int32_t>(*pooledHeightAttr);
        }
    }
    if (attrs != nullptr) {
        const int64_t* pooledWidthAttr = attrs->GetInt(1);
        if (pooledWidthAttr != nullptr) {
            pooledWidth = static_cast<int32_t>(*pooledWidthAttr);
        }
    }
    if(pooledHeight <= 0 || pooledWidth <= 0) {
        OP_LOGE(context, "Invalid pooled height or width");
        return ge::GRAPH_FAILED;
    }
    if (attrs != nullptr) {
        const float* spatialScaleAttr = attrs->GetFloat(2);
        if (spatialScaleAttr != nullptr) {
            spatialScale = *spatialScaleAttr;
        }
    }
    if (attrs != nullptr) {
        const int64_t* samplingRatioAttr = attrs->GetInt(3);
        if (samplingRatioAttr != nullptr) {
            samplingRatio = static_cast<int32_t>(*samplingRatioAttr);
        }
    }
    // Core allocation: distribute ROIs across cores
    uint32_t nowCoreNum = (coreNum > numRois) ? numRois : coreNum;
    OP_CHECK_IF(nowCoreNum == 0, OP_LOGE(context, "nowCoreNum is 0, invalid core number for ROI align"),
        return ge::GRAPH_FAILED);
    uint32_t baseRoisPerCore = numRois / nowCoreNum;
    uint32_t tailRoiNum = numRois % nowCoreNum;
    uint32_t bigTotalRois = baseRoisPerCore + 1;

    // Calculate buffer sizes
    uint32_t featureTotalSize = batch * channels * height * width;
    uint32_t featureMapSize = channels * height * width;
    uint32_t outRoiSize = channels * pooledHeight * pooledWidth;

    tiling->baseRoisPerCore = baseRoisPerCore;
    tiling->bigTotalRois = bigTotalRois;
    tiling->tailRoiNum = tailRoiNum;
    tiling->featureTotalSize = featureTotalSize;
    tiling->roiLength = roiLength;
    tiling->featureMapSize = featureMapSize;
    tiling->outRoiSize = outRoiSize;
    tiling->numRois = numRois;
    tiling->batch = batch;
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->pooledHeight = pooledHeight;
    tiling->pooledWidth = pooledWidth;
    tiling->spatialScale = spatialScale;
    tiling->samplingRatio = samplingRatio;
    context->SetBlockDim(nowCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRoiAlignV2([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(RoiAlignV2).Tiling(RoiAlignV2TilingFunc).TilingParse<RoiAlignV2CompileInfo>(TilingParseForRoiAlignV2);
} // namespace optiling
 