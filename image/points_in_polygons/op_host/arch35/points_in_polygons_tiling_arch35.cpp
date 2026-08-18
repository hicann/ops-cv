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
 * \file points_in_polygons_tiling_arch35.cpp
 * \brief PointsInPolygons host-side tiling for arch35 (Ascend 950)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"

#include "points_in_polygons_tiling_arch35.h"

namespace optiling {

static ge::graphStatus TilingFuncPointsInPolygons(gert::TilingContext* context)
{
    const char* opName = "PointsInPolygons";

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC plat(platformInfo);
    uint32_t coreNum = plat.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::StorageShape* pointsShp = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pointsShp);
    const gert::StorageShape* polygonsShp = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, polygonsShp);
    const auto* pointsDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pointsDesc);
    const auto* polygonsDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, polygonsDesc);

    ge::DataType pointsDtype = pointsDesc->GetDataType();
    ge::DataType polygonsDtype = polygonsDesc->GetDataType();
    if (pointsDtype != ge::DT_FLOAT || polygonsDtype != ge::DT_FLOAT) {
        OP_LOGE(opName, "dtype must be float32, got points=%d, polygons=%d", static_cast<int32_t>(pointsDtype),
                static_cast<int32_t>(polygonsDtype));
        return ge::GRAPH_FAILED;
    }

    ge::Format pointsFmt = pointsDesc->GetStorageFormat();
    ge::Format polygonsFmt = polygonsDesc->GetStorageFormat();
    ge::Format outputFmt = ge::FORMAT_ND;
    const auto* outputDesc = context->GetOutputDesc(0);
    if (outputDesc != nullptr) {
        outputFmt = outputDesc->GetStorageFormat();
    }
    if (pointsFmt != ge::FORMAT_ND || polygonsFmt != ge::FORMAT_ND || outputFmt != ge::FORMAT_ND) {
        OP_LOGE(opName, "format must be ND, got points=%d, polygons=%d, output=%d", static_cast<int32_t>(pointsFmt),
                static_cast<int32_t>(polygonsFmt), static_cast<int32_t>(outputFmt));
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& ptsDims = pointsShp->GetStorageShape();
    const gert::Shape& plyDims = polygonsShp->GetStorageShape();
    if (ptsDims.GetDimNum() != 2 || plyDims.GetDimNum() != 2) {
        OP_LOGE(opName, "shape_mismatch: rank must be 2, got points=%zu, polygons=%zu", ptsDims.GetDimNum(),
                plyDims.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t N = ptsDims.GetDim(0);
    int64_t M = plyDims.GetDim(1);

    bool isEmpty = (N == 0 || M == 0);

    // N-vec 分支：M 较小且 N 足够大时沿 N 轴向量化，避免沿 M 向量化时 VL lane 浪费
    constexpr int64_t N_VEC_M_THRESHOLD = 16;
    constexpr int64_t N_VEC_N_THRESHOLD = 256;
    bool useNVec = !isEmpty && (M <= N_VEC_M_THRESHOLD) && (N >= N_VEC_N_THRESHOLD);

    if (!isEmpty) {
        if (ptsDims.GetDim(1) != 2) {
            OP_LOGE(opName, "shape_mismatch: points.shape[1] must be 2, got %ld", ptsDims.GetDim(1));
            return ge::GRAPH_FAILED;
        }
        if (plyDims.GetDim(0) != 8) {
            OP_LOGE(opName, "shape_mismatch: polygons.shape[0] must be 8, got %ld", plyDims.GetDim(0));
            return ge::GRAPH_FAILED;
        }
    }

    int64_t tileN = 0;
    int64_t tileM = 0;
    int64_t tileNTail = 0;
    int64_t tileMTail = 0;
    int64_t numTilesN = 0;
    int64_t numTilesM = 0;
    uint64_t totalTiles = 0;
    uint64_t perCoreCount = 0;
    uint64_t realCoreNum = 0;
    uint32_t bufferSize = 0;
    uint32_t launchCoreNum = 1;
    int64_t tileNVec = 0;
    int64_t numTilesNVec = 0;
    uint32_t branchKey = POINTS_IN_POLYGONS_KEY_EMPTY;

    if (isEmpty) {
        Branch0Inputs bin{N, M};
        Branch0Outputs bout{};
        ComputeBranch0Tiling(bin, bout);
        tileN = bout.tileN;
        tileM = bout.tileM;
        tileNTail = bout.tileNTail;
        tileMTail = bout.tileMTail;
        numTilesN = bout.numTilesN;
        numTilesM = bout.numTilesM;
        totalTiles = bout.totalTiles;
        perCoreCount = bout.perCoreCount;
        realCoreNum = bout.realCoreNum;
        bufferSize = bout.bufferSize;
        launchCoreNum = bout.launchCoreNum;
        branchKey = POINTS_IN_POLYGONS_KEY_EMPTY;
    } else if (useNVec) {
        Branch2Inputs bin{N, M, coreNum, ubSize};
        Branch2Outputs bout{};
        ComputeBranch2Tiling(bin, bout);
        tileNVec = bout.tileNVec;
        numTilesNVec = bout.numTilesNVec;
        totalTiles = bout.totalTiles;
        perCoreCount = bout.perCoreCount;
        realCoreNum = bout.realCoreNum;
        bufferSize = bout.bufferSize;
        launchCoreNum = bout.launchCoreNum;
        branchKey = POINTS_IN_POLYGONS_KEY_N_VEC;
    } else {
        Branch1Inputs bin{N, M, coreNum, ubSize};
        Branch1Outputs bout{};
        ComputeBranch1Tiling(bin, bout);
        tileN = bout.tileN;
        tileM = bout.tileM;
        tileNTail = bout.tileNTail;
        tileMTail = bout.tileMTail;
        numTilesN = bout.numTilesN;
        numTilesM = bout.numTilesM;
        totalTiles = bout.totalTiles;
        perCoreCount = bout.perCoreCount;
        realCoreNum = bout.realCoreNum;
        bufferSize = bout.bufferSize;
        launchCoreNum = bout.launchCoreNum;
        branchKey = POINTS_IN_POLYGONS_KEY_NORMAL;
    }

    PointsInPolygonsTilingData* td = context->GetTilingData<PointsInPolygonsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);

    td->outN = N;
    td->outM = M;
    td->tileN = tileN;
    td->tileM = tileM;
    td->tileNTail = tileNTail;
    td->tileMTail = tileMTail;
    td->numTilesN = numTilesN;
    td->numTilesM = numTilesM;
    td->totalTiles = totalTiles;
    td->perCoreCount = perCoreCount;
    td->realCoreNum = realCoreNum;
    td->bufferSize = bufferSize;
    td->tileNVec = tileNVec;
    td->numTilesNVec = numTilesNVec;

    context->SetBlockDim(launchCoreNum);

    if (branchKey == POINTS_IN_POLYGONS_KEY_EMPTY) {
        context->SetTilingKey(GET_TPL_TILING_KEY(POINTS_IN_POLYGONS_KEY_EMPTY));
    } else if (branchKey == POINTS_IN_POLYGONS_KEY_N_VEC) {
        context->SetTilingKey(GET_TPL_TILING_KEY(POINTS_IN_POLYGONS_KEY_N_VEC));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(POINTS_IN_POLYGONS_KEY_NORMAL));
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    OP_LOGI(opName, "N=%lld, M=%lld, isEmpty=%d, useNVec=%d, tilingKey=%u, coreNum=%u, ubSize=%llu",
            static_cast<long long>(N), static_cast<long long>(M), static_cast<int32_t>(isEmpty),
            static_cast<int32_t>(useNVec), branchKey, coreNum, static_cast<unsigned long long>(ubSize));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForPointsInPolygons(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto compileInfo = context->GetCompiledInfo<PointsInPolygonsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

void ComputeBranch0Tiling(const Branch0Inputs& in, Branch0Outputs& out)
{
    (void)in;
    out.tileN = 0;
    out.tileM = 0;
    out.tileNTail = 0;
    out.tileMTail = 0;
    out.numTilesN = 0;
    out.numTilesM = 0;
    out.totalTiles = 0;
    out.perCoreCount = 0;
    out.realCoreNum = 0;
    out.bufferSize = 0;
    out.launchCoreNum = 1;
    out.tilingKey = POINTS_IN_POLYGONS_KEY_EMPTY;
}

static inline int64_t CeilDivI64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// UB 双轴反推：固定 tileM，由单 tile UB 上界反解 tileN
static void FindTileNM(uint64_t ubAvailableSingle, uint32_t sizeofT, uint32_t alignElems, int64_t* tileN,
                       int64_t* tileM)
{
    const int64_t coeffB0PerTileN = 2 * static_cast<int64_t>(sizeofT);
    const int64_t coeffB1PerTileM = 8 * static_cast<int64_t>(sizeofT);
    const int64_t coeffCalcPerNm = 4 * static_cast<int64_t>(sizeofT);

    int64_t mUpper = static_cast<int64_t>(ubAvailableSingle / (4 * 8 * sizeofT));
    int64_t candidateM = 32;
    if (candidateM > mUpper) {
        candidateM = mUpper;
    }
    if (candidateM < 1) {
        candidateM = 1;
    }
    candidateM = (candidateM / static_cast<int64_t>(alignElems)) * static_cast<int64_t>(alignElems);
    if (candidateM < 1) {
        candidateM = 1;
    }
    *tileM = candidateM;

    int64_t denom = coeffB0PerTileN + coeffCalcPerNm * (*tileM);
    int64_t numer = static_cast<int64_t>(ubAvailableSingle) - coeffB1PerTileM * (*tileM);
    int64_t candidateN = numer / denom;
    if (candidateN < 1) {
        candidateN = 1;
    }
    candidateN = (candidateN / static_cast<int64_t>(alignElems)) * static_cast<int64_t>(alignElems);
    if (candidateN < 1) {
        candidateN = 1;
    }
    *tileN = candidateN;
}

static void MultiCoreSplit(uint64_t totalTiles, uint32_t coreNum, uint64_t* perCoreCount, uint64_t* realCoreNum)
{
    if (totalTiles == 0) {
        *perCoreCount = 0;
        *realCoreNum = 0;
        return;
    }
    uint64_t coreNumU = static_cast<uint64_t>(coreNum);
    if (coreNumU == 0) {
        coreNumU = 1;
    }
    *perCoreCount = CeilDivU64(totalTiles, coreNumU);
    if (*perCoreCount == 0) {
        *perCoreCount = 1;
    }
    *realCoreNum = CeilDivU64(totalTiles, *perCoreCount);
}

void ComputeBranch1Tiling(const Branch1Inputs& in, Branch1Outputs& out)
{
    constexpr uint32_t SIZEOF_FLOAT = 4;
    constexpr uint32_t ALIGN_ELEMS = 8;

    uint64_t ubAvailableSingle = in.ubSize / 2;
    FindTileNM(ubAvailableSingle, SIZEOF_FLOAT, ALIGN_ELEMS, &out.tileN, &out.tileM);

    out.tileNTail = in.N % out.tileN;
    out.tileMTail = in.M % out.tileM;
    out.numTilesN = CeilDivI64(in.N, out.tileN);
    out.numTilesM = CeilDivI64(in.M, out.tileM);

    out.totalTiles = static_cast<uint64_t>(out.numTilesN) * static_cast<uint64_t>(out.numTilesM);
    MultiCoreSplit(out.totalTiles, in.coreNum, &out.perCoreCount, &out.realCoreNum);

    out.bufferSize = static_cast<uint32_t>(SIZEOF_FLOAT * out.tileN * out.tileM);
    out.launchCoreNum = static_cast<uint32_t>(out.realCoreNum);
    out.tilingKey = POINTS_IN_POLYGONS_KEY_NORMAL;
}

void ComputeBranch2Tiling(const Branch2Inputs& in, Branch2Outputs& out)
{
    constexpr uint32_t SIZEOF_FLOAT = 4;
    constexpr uint32_t ALIGN_ELEMS = 8;

    uint64_t ubAvailableSingle = in.ubSize / 2;
    int64_t fixedOverhead = in.M * 8 * SIZEOF_FLOAT;
    int64_t perTileN = (2 + in.M) * SIZEOF_FLOAT;
    int64_t numer = static_cast<int64_t>(ubAvailableSingle) - fixedOverhead;
    int64_t candidateN = numer / perTileN;
    if (candidateN < 1) {
        candidateN = 1;
    }
    candidateN = (candidateN / static_cast<int64_t>(ALIGN_ELEMS)) * static_cast<int64_t>(ALIGN_ELEMS);
    if (candidateN < 1) {
        candidateN = 1;
    }
    out.tileNVec = candidateN;

    out.numTilesNVec = CeilDivI64(in.N, out.tileNVec);

    out.totalTiles = static_cast<uint64_t>(out.numTilesNVec);
    MultiCoreSplit(out.totalTiles, in.coreNum, &out.perCoreCount, &out.realCoreNum);

    out.bufferSize = static_cast<uint32_t>(SIZEOF_FLOAT * out.tileNVec * in.M);
    out.launchCoreNum = static_cast<uint32_t>(out.realCoreNum);
    out.tilingKey = POINTS_IN_POLYGONS_KEY_N_VEC;
}

IMPL_OP_OPTILING(PointsInPolygons)
    .Tiling(TilingFuncPointsInPolygons)
    .TilingParse<PointsInPolygonsCompileInfo>(TilingPrepareForPointsInPolygons);

} // namespace optiling
