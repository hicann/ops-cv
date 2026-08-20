/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_TILING_DATA_H_
#define GAUSSIAN_BLUR_TILING_DATA_H_

#include <cstdint>

static constexpr uint32_t GAUSSIAN_BLUR_KERNEL_MAX_SIZE = 31U;
static constexpr uint32_t GAUSSIAN_BLUR_CHANNEL_TILE = 4U;
static constexpr uint32_t GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP = 8U;
static constexpr uint32_t GAUSSIAN_BLUR_THREADS = 256U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_BLOCK_X = 32U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_BLOCK_Y = 8U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_PATCHES = 4U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_TILE_W = 128U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_TILE_H = 8U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_UB_PATCH_W = 192U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS = 4U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES = GAUSSIAN_BLUR_ROW_TILE_H * GAUSSIAN_BLUR_ROW_UB_PATCH_W *
                                                              GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS * sizeof(float);
static constexpr uint32_t GAUSSIAN_BLUR_COLUMN_BLOCK_X = 16U;
static constexpr uint32_t GAUSSIAN_BLUR_COLUMN_BLOCK_Y = 16U;
static constexpr uint32_t GAUSSIAN_BLUR_COLUMN_PATCHES = 6U;
static constexpr uint32_t GAUSSIAN_BLUR_COLUMN_TILE_W = 16U;
static constexpr uint32_t GAUSSIAN_BLUR_COLUMN_TILE_H = 96U;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_SHARED_ELEMENTS = GAUSSIAN_BLUR_ROW_TILE_H *
                                                              (GAUSSIAN_BLUR_ROW_PATCHES + 2U) *
                                                              GAUSSIAN_BLUR_ROW_BLOCK_X *
                                                              GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS;
static constexpr uint32_t GAUSSIAN_BLUR_ROW_SHARED_UB_BYTES = GAUSSIAN_BLUR_ROW_SHARED_ELEMENTS * sizeof(float);
static constexpr uint32_t GAUSSIAN_BLUR_PATH_GENERIC_C = 0U;
static constexpr uint32_t GAUSSIAN_BLUR_PATH_C1_FAST = 1U;
static constexpr uint32_t GAUSSIAN_BLUR_PATH_C3_FAST = 2U;
static constexpr uint32_t GAUSSIAN_BLUR_PATH_C4_FAST = 3U;
static constexpr uint32_t GAUSSIAN_BLUR_PATH_GENERIC_C8 = 4U;
static constexpr uint32_t GAUSSIAN_BLUR_PADDING_CONSTANT = 0U;
static constexpr uint32_t GAUSSIAN_BLUR_PADDING_REPLICATE = 1U;
static constexpr uint32_t GAUSSIAN_BLUR_PADDING_REFLECT_101 = 4U;

struct GaussianBlurTilingData {
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t totalTiles;
    uint32_t tilesX;
    uint32_t tilesY;
    uint32_t kernelSize;
    uint32_t radius;
    uint32_t borderType;
    uint32_t pathMode;
    uint32_t kernelSizeY;
    uint32_t reserved[2];
    float weights[GAUSSIAN_BLUR_KERNEL_MAX_SIZE];
    float weightsY[GAUSSIAN_BLUR_KERNEL_MAX_SIZE];
};

#endif // GAUSSIAN_BLUR_TILING_DATA_H_
