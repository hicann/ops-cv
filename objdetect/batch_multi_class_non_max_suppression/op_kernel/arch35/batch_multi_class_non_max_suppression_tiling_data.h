/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_DATA_H_
#define BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_DATA_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct BatchMultiClassNonMaxSuppressionTilingData {
    int64_t batch{0};
    int64_t boxesNum{0};
    int64_t classesNum{0};
    int64_t boxClassesNum{0};
    int64_t maxSizePerClass{0};
    int64_t maxTotalSize{0};
    int64_t usedCoreNum{0};
    int64_t tileSize{0};
    int64_t reduceBufferSize{0};
    int64_t mergeInputCount{0};
    int64_t mergeInputSize{0};
    int64_t mergeOutputCount{0};
    int64_t mergeOutputSize{0};
    uint64_t scratchFieldStride{0};
    uint64_t scratchBytesPerCore{0};
    uint64_t classBoxesOffset{0};
    uint64_t classScoresOffset{0};
    uint64_t classCountsOffset{0};
    uint64_t mergeScoresOffset{0};
    uint64_t mergeIndicesOffset{0};
    uint64_t topKTempBytes{0};
    AscendC::tiling::TopkTiling mergeTopKTiling{};
    float scoreThreshold{0.0F};
    float iouThreshold{0.0F};
    uint8_t hasClipWindow{0};
    uint8_t hasNumValidBoxes{0};
    uint8_t changeCoordinateFrame{0};
    uint8_t transposeBox{0};
    uint8_t use32Index{0};
};

#endif // BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_DATA_H_
