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
 * \file points_in_polygons_tiling_data.h
 * \brief PointsInPolygons TilingData POD struct
 */

#ifndef POINTS_IN_POLYGONS_TILING_DATA_H_
#define POINTS_IN_POLYGONS_TILING_DATA_H_
#include <cstdint>

struct PointsInPolygonsTilingData {
    int64_t outN; // N dimension
    int64_t outM; // M dimension

    int64_t tileN;     // N-axis tile size
    int64_t tileM;     // M-axis tile size
    int64_t tileNTail; // N % tileN
    int64_t tileMTail; // M % tileM
    int64_t numTilesN; // CeilDiv(N, tileN)
    int64_t numTilesM; // CeilDiv(M, tileM)

    uint64_t totalTiles;   // numTilesN * numTilesM
    uint64_t perCoreCount; // CeilDiv(totalTiles, coreNum)
    uint64_t realCoreNum;  // CeilDiv(totalTiles, perCoreCount)

    uint32_t bufferSize; // single-tile bytes (excl. ping-pong)

    // N-vec branch (tilingKey=2): vectorize along N axis
    int64_t tileNVec;     // N-axis tile size in N-vec mode
    int64_t numTilesNVec; // CeilDiv(N, tileNVec)
};

#endif // POINTS_IN_POLYGONS_TILING_DATA_H_
