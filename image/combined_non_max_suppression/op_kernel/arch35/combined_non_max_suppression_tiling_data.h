/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMBINED_NON_MAX_SUPPRESSION_TILING_DATA_H_
#define COMBINED_NON_MAX_SUPPRESSION_TILING_DATA_H_

#include <cstdint>

struct CombinedNonMaxSuppressionTilingData {
    int32_t batchSize;
    int32_t numBoxes;
    int32_t boxClasses;
    int32_t numClasses;
    int32_t maxOutputPerClass;
    int32_t maxTotalSize;
    int32_t outputSize;
    int32_t usedCoreNum;
    int32_t clipBoxes;
    float iouThreshold;
    float scoreThreshold;
    uint64_t selectedScoresOffset;
    uint64_t selectedIndicesOffset;
    uint64_t selectedCountsOffset;
    uint64_t suppressedOffset;
};

#endif // COMBINED_NON_MAX_SUPPRESSION_TILING_DATA_H_
