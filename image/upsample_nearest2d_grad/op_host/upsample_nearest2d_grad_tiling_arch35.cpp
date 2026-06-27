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
 * \file upsample_nearest2d_grad_tiling.cpp
 * \brief Tiling implementation for upsample_nearest2d_grad operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/upsample_nearest2d_grad_tiling_data.h"
#include "../op_kernel/arch35/upsample_nearest2d_grad_tiling_key.h"

namespace optiling {

constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr int32_t INPUT_SIZE_ATTR_LEN = 4;
constexpr int32_t OUTPUT_SIZE_ATTR_LEN = 2;

struct UpsampleNearest2dGradCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingAttrs(gert::TilingContext* context, int32_t& dimN, int32_t& dimC, int32_t& dimHin,
                                      int32_t& dimWin, int32_t& dimHout, int32_t& dimWout, float& scaleH, float& scaleW)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Attr 0: output_size (ListInt) [H_out, W_out]
    const auto* outputSizeVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSizeVec);
    if (outputSizeVec->GetSize() != OUTPUT_SIZE_ATTR_LEN) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "output_size",
                                     std::to_string(outputSizeVec->GetSize()).c_str(),
                                     std::to_string(OUTPUT_SIZE_ATTR_LEN).c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* outputSizeData = outputSizeVec->GetData();
    dimHout = static_cast<int32_t>(outputSizeData[0]);
    dimWout = static_cast<int32_t>(outputSizeData[1]);

    // Attr 1: input_size (ListInt) [N, C, H_in, W_in]
    const auto* inputSizeVec = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeVec);
    if (inputSizeVec->GetSize() != INPUT_SIZE_ATTR_LEN) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "input_size",
                                     std::to_string(inputSizeVec->GetSize()).c_str(),
                                     std::to_string(INPUT_SIZE_ATTR_LEN).c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* inputSizeData = inputSizeVec->GetData();
    dimN = static_cast<int32_t>(inputSizeData[0]);
    dimC = static_cast<int32_t>(inputSizeData[1]);
    dimHin = static_cast<int32_t>(inputSizeData[2]);
    dimWin = static_cast<int32_t>(inputSizeData[3]);

    if (dimN <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_size", std::to_string(dimN).c_str(),
                                              "dimN must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dimC <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_size", std::to_string(dimC).c_str(),
                                              "dimC must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dimHin <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_size", std::to_string(dimHin).c_str(),
                                              "dimHin must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dimWin <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_size", std::to_string(dimWin).c_str(),
                                              "dimWin must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dimHout <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size", std::to_string(dimHout).c_str(),
                                              "dimHout must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dimWout <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size", std::to_string(dimWout).c_str(),
                                              "dimWout must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    // Attr 2: scales_h (Float, default 0.0)
    float scalesHVal = 0.0f;
    const float* scalesHPtr = attrs->GetFloat(2);
    if (scalesHPtr != nullptr) {
        scalesHVal = *scalesHPtr;
    }

    // Attr 3: scales_w (Float, default 0.0)
    float scalesWVal = 0.0f;
    const float* scalesWPtr = attrs->GetFloat(3);
    if (scalesWPtr != nullptr) {
        scalesWVal = *scalesWPtr;
    }

    // Compute scale
    scaleH = (scalesHVal > 0.0f) ? scalesHVal : static_cast<float>(dimHout) / static_cast<float>(dimHin);
    scaleW = (scalesWVal > 0.0f) ? scalesWVal : static_cast<float>(dimWout) / static_cast<float>(dimWin);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest2dGradArch35TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int32_t dimN = 0, dimC = 0, dimHin = 0, dimWin = 0, dimHout = 0, dimWout = 0;
    float scaleH = 0.0f, scaleW = 0.0f;
    OP_CHECK_IF(
        GetTilingAttrs(context, dimN, dimC, dimHin, dimWin, dimHout, dimWout, scaleH, scaleW) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetTilingAttrs error"), return ge::GRAPH_FAILED);

    // Compute tiling parameters
    int64_t totalElements = (int64_t)dimC * dimHin * dimWin;

    int64_t perCoreElements = (totalElements > 0) ? Ops::Base::CeilDiv(totalElements, coreNum) : 0;
    if (perCoreElements > 0 && perCoreElements < PER_CORE_MIN_ELEMENTS) {
        perCoreElements = PER_CORE_MIN_ELEMENTS;
    }
    // Align to 32 (warp size)
    if (perCoreElements > 0) {
        perCoreElements = ((perCoreElements + 31) / 32) * 32;
    }
    int32_t needCoreNum = (totalElements > 0) ?
                              static_cast<int32_t>(Ops::Base::CeilDiv(totalElements, perCoreElements)) :
                              1;
    if (needCoreNum < 1) {
        needCoreNum = 1;
    }

    // Fill tiling data
    UpsampleNearest2dGradTilingData* tiling = context->GetTilingData<UpsampleNearest2dGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(UpsampleNearest2dGradTilingData), 0, sizeof(UpsampleNearest2dGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->needCoreNum = needCoreNum;
    tiling->totalElements = totalElements;
    tiling->dimN = dimN;
    tiling->dimC = dimC;
    tiling->dimHin = dimHin;
    tiling->dimWin = dimWin;
    tiling->dimHout = dimHout;
    tiling->dimWout = dimWout;
    tiling->scaleH = scaleH;
    tiling->scaleW = scaleW;

    context->SetBlockDim(needCoreNum);

    // Set tiling key (single mode, dtype by DTYPE_ macro)
    uint64_t tilingKey = GET_TPL_TILING_KEY(UPSAMPLE_NEAREST2D_GRAD_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    // Set workspace (system workspace only)
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);

    // Set local memory size
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
