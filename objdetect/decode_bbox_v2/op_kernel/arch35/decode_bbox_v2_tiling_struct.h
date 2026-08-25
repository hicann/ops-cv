/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DECODE_BBOX_V2_TILING_STRUCT_H
#define DECODE_BBOX_V2_TILING_STRUCT_H

#include <cstdint>

constexpr int64_t DECODE_BBOX_V2_ELEMS_PER_BOX = 4;
constexpr int64_t DECODE_BBOX_V2_MIN_TILING_BITS = 32768;
constexpr int64_t DECODE_BBOX_V2_ELEM_ALIGN_FACTOR = 512;
constexpr int64_t DECODE_BBOX_V2_ALIGN_256 = 256;
constexpr int64_t DECODE_BBOX_V2_RESERVED_UB = 20480;

struct DecodeBboxV2TilingData {
    int64_t dim0;
    int32_t coreNum;
    int64_t blockFormer;
    int64_t blockNum;

    int64_t ubFormer;
    int64_t ubLoopOfFormerBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfTailBlock;

    float scales[4];
    float decodeClip;
    float invScales[4];
    float halfVal;
};

#endif
