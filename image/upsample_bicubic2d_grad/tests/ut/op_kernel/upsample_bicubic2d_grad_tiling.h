/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __BIDIRECTION_LSTM_TILING_H__
#define __BIDIRECTION_LSTM_TILING_H__

#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_KT_TEST__)
#include <cstdint>
#include <cstring>
#endif

constexpr uint16_t MAX_CORE_COUNT = 50;

#pragma pack(1)
struct UpsampleBicubic2dGradTilingData {
    uint32_t dataType = 0;
    uint32_t CoreNum = 0;
    uint32_t alignCorners = 0;
    float scalesH = 0;
    float scalesW = 0;
    uint32_t baseNH = 0;
    uint32_t baseNW = 0;
    uint32_t batch = 0;
    uint32_t inputN = 0;
    uint32_t inputC = 0;
    uint32_t inputH = 0;
    uint32_t inputW = 0;
    uint32_t outputH = 0;
    uint32_t outputW = 0;
    uint32_t tailH = 0;
    uint32_t CoreNumH = 0;
    uint32_t loopH = 0;
    uint32_t loopTailCoreH = 0;
    uint32_t innerCoreNumH = 0;
    uint32_t innerBatchH = 0;
    uint32_t innerBatchTailCoreH = 0;
    uint32_t tailW = 0;
    uint32_t CoreNumW = 0;
    uint32_t loopW = 0;
    uint32_t loopTailCoreW = 0;
    uint32_t innerCoreNumW = 0;
    uint32_t innerBatchW = 0;
    uint32_t innerBatchTailCoreW = 0;
    uint32_t clearBaseN = 0;
    uint32_t clearInterLoop = 0;
    uint32_t clearInterTailN = 0;
    uint32_t clearInterTailCoreNum = 0;
    uint32_t clearOutLoop = 0;
    uint32_t clearOutTailN = 0;
    uint32_t clearOutTailCoreNum = 0;

    uint32_t slideSize = 0;
    uint32_t needExpandW = 0;
    uint32_t needExpandH = 0;

    uint32_t tailStartW = 0;
    uint32_t tailEndW = 0;
    uint32_t tailStartH = 0;
    uint32_t tailEndH = 0;

    uint64_t singleCoreKW = 0;
    uint64_t singleCoreKH = 0;
    uint64_t radioMatrixSize = 0;
    uint64_t intermediateMatrixSize = 0;

    uint32_t slideStartListW[MAX_CORE_COUNT] = {0};
    uint32_t slideEndListW[MAX_CORE_COUNT] = {0};
    uint32_t tailSlideStartListW[MAX_CORE_COUNT] = {0};
    uint32_t tailSlideEndListW[MAX_CORE_COUNT] = {0};

    uint32_t slideStartListH[MAX_CORE_COUNT] = {0};
    uint32_t slideEndListH[MAX_CORE_COUNT] = {0};
    uint32_t tailSlideStartListH[MAX_CORE_COUNT] = {0};
    uint32_t tailSlideEndListH[MAX_CORE_COUNT] = {0};

    TCubeTiling MMParamH;
    TCubeTiling MMParamW;
};
#pragma pack()

#if defined(__CCE_KT_TEST__)
inline void InitTilingData(uint8_t *tiling, UpsampleBicubic2dGradTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(UpsampleBicubic2dGradTilingData));
}
#else
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, UpsampleBicubic2dGradTilingData *const_data)
{
    for (auto i = 0; i < sizeof(UpsampleBicubic2dGradTilingData) / 4; i++) {
        *(int32_t *)((int32_t *)const_data + i) = *((__gm__ int32_t *)tiling + i);
    }
}
#endif

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    UpsampleBicubic2dGradTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)
#endif
