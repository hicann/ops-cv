/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
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
 * \file nms_with_mask_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "experimental/image/nms_with_mask/op_kernel/nms_with_mask_tiling_data.h"
#include "experimental/image/nms_with_mask/op_kernel/nms_with_mask_tiling_key.h"

namespace optiling {

using namespace Ops::Cv::OpTiling;

constexpr uint32_t CoreNum = 1;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int32_t DIMS_LIMIT_0 = 2;
constexpr int32_t DIMS_LIMIT_1 = 1;
constexpr int32_t LIMIT_INPUT_NUM = 2048 * 4;//2048个框
constexpr int32_t ATTRPOS0 = 0;
constexpr int32_t ATTRPOS1 = 1;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;

struct NMSWithMaskCompileInfo {};

// 获取平台信息如ubSize, coreNum
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

// 获取属性，shape信息
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());

    // shape校验
    OP_CHECK_IF(
        inputShapeX.GetDimNum() != DIMS_LIMIT_0 || inputShapeY.GetDimNum() != DIMS_LIMIT_1 ||
            outShapeZ.GetDimNum() != DIMS_LIMIT_1,
        OP_LOGE(
            context, "NMSWithMask: inputx,inputy,outputz shape dim = %zu, %zu, %zu, should be equal 2,1,1",
            inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
        return ge::GRAPH_FAILED);

    // 获取shape dim值
    auto nDim = inputShapeX.GetDim(INDEXZERO);
    auto cDim = inputShapeX.GetDim(INDEXONE);
    totalIdx = nDim * cDim;
    if(totalIdx > LIMIT_INPUT_NUM){
        OP_LOGE(context, "invalid Input");
        return ge::GRAPH_FAILED;
    }
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus NMSWithMaskTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    int64_t totalIdx;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    NMSWithMaskTilingData* tiling = context->GetTilingData<NMSWithMaskTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(NMSWithMaskTilingData), 0, sizeof(NMSWithMaskTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = totalIdx;
    tiling->iou_threshold = *context->GetAttrs()->GetAttrPointer<float>(ATTRPOS0);
    tiling->scores_threshold = *context->GetAttrs()->GetAttrPointer<float>(ATTRPOS1); 

    context->SetBlockDim(CoreNum);
    uint64_t tilingKey = 0;
    // 区分dtype走不同得tiling key分支.
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNMSWithMask([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(NMSWithMask).Tiling(NMSWithMaskTilingFunc).TilingParse<NMSWithMaskCompileInfo>(TilingParseForNMSWithMask);
} // namespace optiling
