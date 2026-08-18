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
 * \file points_in_polygons_tiling_arch35.h
 * \brief PointsInPolygons tiling compile info and branch I/O structs for arch35
 */

#ifndef OPS_POINTS_IN_POLYGONS_OP_HOST_ARCH35_TILING_ARCH35_H
#define OPS_POINTS_IN_POLYGONS_OP_HOST_ARCH35_TILING_ARCH35_H

#include "../../op_kernel/arch35/points_in_polygons_tiling_data.h"
#include "../../op_kernel/arch35/points_in_polygons_tiling_key.h"

namespace optiling {

struct PointsInPolygonsCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

// Branch-0 (empty / tilingKey=0): empty tensor short-circuit
struct Branch0Inputs {
    int64_t N;
    int64_t M;
};

struct Branch0Outputs {
    int64_t tileN;
    int64_t tileM;
    int64_t tileNTail;
    int64_t tileMTail;
    int64_t numTilesN;
    int64_t numTilesM;
    uint64_t totalTiles;
    uint64_t perCoreCount;
    uint64_t realCoreNum;
    uint32_t bufferSize;
    uint32_t launchCoreNum;
    uint32_t tilingKey;
};

void ComputeBranch0Tiling(const Branch0Inputs& in, Branch0Outputs& out);

// Branch-1 (normal / tilingKey=1): 2D tile + multi-core split
struct Branch1Inputs {
    int64_t N;
    int64_t M;
    uint32_t coreNum;
    uint64_t ubSize;
};

struct Branch1Outputs {
    int64_t tileN;
    int64_t tileM;
    int64_t tileNTail;
    int64_t tileMTail;
    int64_t numTilesN;
    int64_t numTilesM;
    uint64_t totalTiles;
    uint64_t perCoreCount;
    uint64_t realCoreNum;
    uint32_t bufferSize;
    uint32_t launchCoreNum;
    uint32_t tilingKey;
};

void ComputeBranch1Tiling(const Branch1Inputs& in, Branch1Outputs& out);

// Branch-2 (N-vec / tilingKey=2): vectorize along N axis
struct Branch2Inputs {
    int64_t N;
    int64_t M;
    uint32_t coreNum;
    uint64_t ubSize;
};

struct Branch2Outputs {
    int64_t tileNVec;
    int64_t numTilesNVec;
    uint64_t totalTiles;
    uint64_t perCoreCount;
    uint64_t realCoreNum;
    uint32_t bufferSize;
    uint32_t launchCoreNum;
    uint32_t tilingKey;
};

void ComputeBranch2Tiling(const Branch2Inputs& in, Branch2Outputs& out);

} // namespace optiling

#endif // OPS_POINTS_IN_POLYGONS_OP_HOST_ARCH35_TILING_ARCH35_H
