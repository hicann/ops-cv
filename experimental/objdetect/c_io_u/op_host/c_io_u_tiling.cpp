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
 * \file c_io_u_tiling.cpp
 * \brief CIoU tiling.  Per-core partition is cache-line aligned to avoid
 *        GM write-back conflicts on the (1, N) outputs.
 */

#include <cstring>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "experimental/objdetect/c_io_u/op_kernel/c_io_u_tiling_data.h"
#include "experimental/objdetect/c_io_u/op_kernel/c_io_u_tiling_key.h"

namespace optiling {

// TILE_N: tested {512, 1024}; 512 wins by ~5% (geomean 6.43x vs 6.10x).
// 1024 didn't help: small-N cases (N=1024) only need 1 iter at tileN=1024 but
// Atan ws + 26 fp32 bufs at 4KB each pressures UB; medium cases (N=2048-8192)
// split inner-loop = N/(48*1024) which is <= 2 iters either way.  The Atan
// setup amortization gain is < the UB-pressure cost.
static constexpr uint32_t TILE_N = 512;

struct CIoUAttrs {
    bool trans = false;
    bool isCross = false;
    const char* modeStr = "iou";
    bool atanSubFlag = true;
};

struct CorePartition {
    uint32_t usedCoreNum = 1;
    uint32_t basePerCore = 0;
    uint32_t pivot = 0;
    uint32_t alignElem = 0;
    uint32_t tailN = 0;
};

static ge::graphStatus InitTilingData(gert::TilingContext* context, CIoUTilingData*& tiling)
{
    tiling = context->GetTilingData<CIoUTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CIoUTilingData), 0, sizeof(CIoUTilingData)) != EOK,
                OP_LOGE(context, "memset CIoUTilingData failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTotalN(gert::TilingContext* context, uint32_t& totalN)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() < 2, OP_LOGE(context, "CIoU: bboxes rank < 2"),
                return ge::GRAPH_FAILED);
    totalN = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static CIoUAttrs ReadAttrs(gert::TilingContext* context)
{
    CIoUAttrs ciouAttrs;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* p;
        if ((p = attrs->GetAttrPointer<bool>(0)) != nullptr)
            ciouAttrs.trans = *p;
        if ((p = attrs->GetAttrPointer<bool>(1)) != nullptr)
            ciouAttrs.isCross = *p;
        const char* s = attrs->GetAttrPointer<char>(2);
        if (s != nullptr)
            ciouAttrs.modeStr = s;
        if ((p = attrs->GetAttrPointer<bool>(3)) != nullptr)
            ciouAttrs.atanSubFlag = *p;
    }
    return ciouAttrs;
}

static ge::graphStatus GetModeId(gert::TilingContext* context, const CIoUAttrs& ciouAttrs, int32_t& modeId)
{
    // is_cross == true (cross-pair MxN output) is NOT supported by this kernel.
    // aclnnCIoU likewise rejects isCross=true at host check.
    OP_CHECK_IF(ciouAttrs.isCross, OP_LOGE(context, "CIoU: is_cross=true is not supported"), return ge::GRAPH_FAILED);

    modeId = 0;
    if (std::strcmp(ciouAttrs.modeStr, "iou") == 0) {
        modeId = 0;
    } else if (std::strcmp(ciouAttrs.modeStr, "iof") == 0) {
        modeId = 1;
    } else {
        OP_LOGE(context, "CIoU: unsupported mode '%s' (expect 'iou' or 'iof')", ciouAttrs.modeStr);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildPartition(gert::TilingContext* context, uint32_t totalN, uint32_t& coreNum,
                                      CorePartition& partition)
{
    auto inDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inDesc);
    const auto outDtype = inDesc->GetDataType();
    uint32_t typeSize = (outDtype == ge::DT_FLOAT) ? 4u : 2u;
    partition.alignElem = 32u / typeSize; // 8 for fp32, 16 for fp16
    uint32_t alignElem = partition.alignElem;
    uint32_t alignedChunks = (alignElem == 0) ? 0u : (totalN / alignElem);
    partition.tailN = (alignElem == 0) ? 0u : (totalN % alignElem);

    auto* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0)
        coreNum = 1;

    if (totalN == 0) {
        partition.tailN = 0;
    } else if (alignedChunks == 0) {
        partition.basePerCore = totalN;
        partition.tailN = 0;
    } else {
        partition.usedCoreNum = (alignedChunks < coreNum) ? alignedChunks : coreNum;
        uint32_t chunksPerCore = alignedChunks / partition.usedCoreNum;
        uint32_t chunkRem = alignedChunks % partition.usedCoreNum;
        partition.basePerCore = chunksPerCore * alignElem;
        partition.pivot = chunkRem;
    }
    return ge::GRAPH_SUCCESS;
}

static void FillTilingData(CIoUTilingData* tiling, uint32_t totalN, const CIoUAttrs& ciouAttrs,
                           const CorePartition& partition, int32_t modeId)
{
    tiling->totalN = totalN;
    tiling->basePerCore = partition.basePerCore;
    tiling->pivot = partition.pivot;
    tiling->tileN = TILE_N;
    tiling->usedCoreNum = partition.usedCoreNum;
    tiling->alignElem = partition.alignElem;
    tiling->tailN = partition.tailN;
    tiling->trans = ciouAttrs.trans ? 1 : 0;
    tiling->modeId = modeId;
    tiling->atanSubFlag = ciouAttrs.atanSubFlag ? 1 : 0;
    tiling->eps = 1e-7f;
}

static ge::graphStatus CIoUTilingFunc(gert::TilingContext* context)
{
    CIoUTilingData* tiling = nullptr;
    uint32_t totalN = 0;
    OP_CHECK_IF(InitTilingData(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CIoU: InitTilingData failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetTotalN(context, totalN) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CIoU: GetTotalN failed"),
                return ge::GRAPH_FAILED);

    const CIoUAttrs ciouAttrs = ReadAttrs(context);
    int32_t modeId = 0;
    OP_CHECK_IF(GetModeId(context, ciouAttrs, modeId) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CIoU: GetModeId failed"),
                return ge::GRAPH_FAILED);

    uint32_t coreNum = 1;
    CorePartition partition;
    OP_CHECK_IF(BuildPartition(context, totalN, coreNum, partition) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CIoU: BuildPartition failed"), return ge::GRAPH_FAILED);
    FillTilingData(tiling, totalN, ciouAttrs, partition, modeId);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(partition.usedCoreNum);
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

struct CIoUCompileInfo {};

static ge::graphStatus TilingParseForCIoU([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CIoU).Tiling(CIoUTilingFunc).TilingParse<CIoUCompileInfo>(TilingParseForCIoU);
} // namespace optiling
