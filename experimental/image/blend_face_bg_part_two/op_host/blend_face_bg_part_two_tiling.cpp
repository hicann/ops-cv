/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file blend_face_bg_part_two_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/image/blend_face_bg_part_two/op_kernel/blend_face_bg_part_two_tiling_data.h"
#include "experimental/image/blend_face_bg_part_two/op_kernel/blend_face_bg_part_two_tiling_key.h"

namespace optiling {

constexpr uint32_t MAX_TILE_SIZE = 2048;
constexpr uint32_t MIN_TILE_SIZE = 32;
constexpr uint32_t UB_RESERVE_BYTES = 8U * 1024U;
// Worst case per element: double-buffered 4 inputs + 1 output,
// 4 fp32 scratch buffers, and uint8 cast buffers.
constexpr uint32_t UB_BYTES_PER_ELEM = 56;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int32_t ATTR_EPSILON_INDEX = 0;
constexpr float DEFAULT_EPSILON = 1e-12f;

struct BlendFaceBgPartTwoCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "[BlendFaceBgPartTwo] coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "[BlendFaceBgPartTwo] ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTotalElems(gert::TilingContext* context, const gert::StorageShape* accFaceShape,
                                     uint32_t& totalElems)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, accFaceShape);

    uint64_t totalElems64 = 1;
    auto dimNum = accFaceShape->GetStorageShape().GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        totalElems64 *= static_cast<uint64_t>(accFaceShape->GetStorageShape().GetDim(i));
    }
    OP_CHECK_IF(totalElems64 > UINT32_MAX, OP_LOGE(context, "[BlendFaceBgPartTwo] totalElems exceeds UINT32_MAX"),
                return ge::GRAPH_FAILED);
    totalElems = static_cast<uint32_t>(totalElems64);
    return ge::GRAPH_SUCCESS;
}

static float GetEpsilon(gert::TilingContext* context)
{
    float epsilon = DEFAULT_EPSILON;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsilonPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON_INDEX);
        if (epsilonPtr != nullptr) {
            epsilon = *epsilonPtr;
        }
    }
    return epsilon;
}

static ge::graphStatus CalcTileSize(gert::TilingContext* context, uint64_t ubSize, uint32_t& tileSize)
{
    uint64_t availableUb = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : 0;
    uint64_t maxElems = availableUb / UB_BYTES_PER_ELEM;
    OP_CHECK_IF(maxElems < MIN_TILE_SIZE, OP_LOGE(context, "[BlendFaceBgPartTwo] UB is too small for minimum tile"),
                return ge::GRAPH_FAILED);

    uint32_t rawTileSize = maxElems > MAX_TILE_SIZE ? MAX_TILE_SIZE : static_cast<uint32_t>(maxElems);
    tileSize = (rawTileSize / MIN_TILE_SIZE) * MIN_TILE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(BlendFaceBgPartTwoTilingData* tiling, uint32_t totalElems, uint32_t baseElems, uint32_t pivot,
                          uint32_t tileSize, float epsilon)
{
    tiling->totalElems = totalElems;
    tiling->baseElems = baseElems;
    tiling->pivot = pivot;
    tiling->tileSize = tileSize;
    tiling->epsilon = epsilon;
}

static ge::graphStatus SetEmptyTiling(gert::TilingContext* context, BlendFaceBgPartTwoTilingData* tiling)
{
    context->SetBlockDim(1);
    SetTilingData(tiling, 0, 0, 0, MAX_TILE_SIZE, DEFAULT_EPSILON);
    return SetWorkspace(context);
}

static ge::graphStatus BlendFaceBgPartTwoTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<BlendFaceBgPartTwoTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint32_t totalElems = 0;
    OP_CHECK_IF(GetTotalElems(context, context->GetInputShape(0), totalElems) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[BlendFaceBgPartTwo] GetTotalElems failed"), return ge::GRAPH_FAILED);

    if (totalElems == 0) {
        return SetEmptyTiling(context, tiling);
    }

    uint64_t ubSize;
    uint32_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[BlendFaceBgPartTwo] GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    uint32_t tileSize = 0;
    OP_CHECK_IF(CalcTileSize(context, ubSize, tileSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[BlendFaceBgPartTwo] CalcTileSize failed"), return ge::GRAPH_FAILED);

    uint32_t usedCoreNum = (totalElems < coreNum) ? totalElems : coreNum;
    context->SetBlockDim(usedCoreNum);
    SetTilingData(tiling, totalElems, totalElems / usedCoreNum, totalElems % usedCoreNum, tileSize,
                  GetEpsilon(context));
    return SetWorkspace(context);
}

static ge::graphStatus TilingParseForBlendFaceBgPartTwo([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlendFaceBgPartTwo)
    .Tiling(BlendFaceBgPartTwoTilingFunc)
    .TilingParse<BlendFaceBgPartTwoCompileInfo>(TilingParseForBlendFaceBgPartTwo);
} // namespace optiling
