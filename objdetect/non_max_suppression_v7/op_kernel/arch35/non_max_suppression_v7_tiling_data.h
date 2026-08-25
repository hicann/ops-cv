/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NMS_V7_KERNEL_TILING_DATA_H_
#define NMS_V7_KERNEL_TILING_DATA_H_

#include <cstdint>

struct NonMaxSuppressionV7TilingData {
    int64_t batch{0};
    int64_t classes{0};
    int64_t boxes{0};
    // maxOutputSize is the first dimension of selected_indices.
    int64_t maxOutputSize{0};
    int64_t maxOutputPerClass{0};
    int64_t usedCoreNum{0};
    int64_t tileSize{0};
    int64_t reduceBufferSize{0};
    uint64_t scratchFieldStride{0};
    uint64_t classIndicesOffset{0};
    uint64_t classCountsOffset{0};
    float iouThreshold{0.0F};
    float scoreThreshold{0.0F};
    uint8_t centerPointBox{0};
    uint8_t hasMax{0};
    uint8_t hasIou{0};
    uint8_t hasScore{0};
    uint8_t hasIndex{0};
    // index_id accepts both [B, C, N, 3] and [B, C, N, 4].
    uint8_t indexWidth{0};
};

#endif
