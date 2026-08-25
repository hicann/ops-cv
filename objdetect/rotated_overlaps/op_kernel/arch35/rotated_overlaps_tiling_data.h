/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_OVERLAPS_TILING_DATA_H_
#define ROTATED_OVERLAPS_TILING_DATA_H_

#include <cstdint>

// The vector kernel allocates 145 working vectors plus eight vectors for the
// non-overlapping strided-copy scratch area.  Keeping the total beside the ABI
// makes the host UB budget auditable.
constexpr uint32_t kRotatedOverlapsFloatVectorCount = 153U;
constexpr uint32_t kRotatedOverlapsMaskReserveBytes = 256U;
constexpr uint32_t kRotatedOverlapsControlReserveBytes = 1024U;
constexpr uint32_t kRotatedOverlapsSimtThreadNum = 256U;
constexpr uint32_t kRotatedOverlapsSimtDataCacheReserveBytes = 32U * 1024U;

struct RotatedOverlapsTilingData {
    uint64_t batch = 0;
    uint64_t numBoxes = 0;
    uint64_t numQueries = 0;
    uint64_t totalRows = 0;
    uint64_t totalPairs = 0;
    uint64_t totalTasks = 0;
    uint64_t tasksPerCore = 0;
    uint32_t usedCoreNum = 0;
    uint32_t tileLen = 0;
    uint32_t tilesPerOuter = 0;
    uint32_t mathTmpBytes = 0;
    uint32_t trans = 0;
    uint32_t use32Bit = 0;
    uint32_t vectorizeBoxes = 0;
    uint32_t usePairParallelSimt = 0;
};

#endif // ROTATED_OVERLAPS_TILING_DATA_H_
