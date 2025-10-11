/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _GRID_SAMPLE_TILING_H_
#define _GRID_SAMPLE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 200

#pragma pack(1)

struct GridSampleTilingDataTest {
    int64_t coreNumVar = 0;
    int64_t inN = 0;
    int64_t inC = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    int64_t alignCorners = 0;
    int64_t channelLast = 0;
    int64_t needCoreNum = 0;
    int64_t preCoreNum = 0;
    int64_t preNumPerCore = 0;
    int64_t postNumPerCore = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    GridSampleTilingDataTest tilingData;                                          \
    INIT_TILING_DATA(GridSampleTilingDataTest, tilingDataPointer, tilingPointer); \
    (tilingData).coreNumVar = tilingDataPointer->coreNumVar;                  \
    (tilingData).inN = tilingDataPointer->inN;                                \
    (tilingData).inC = tilingDataPointer->inC;                                \
    (tilingData).inD = tilingDataPointer->inD;                                \
    (tilingData).inH = tilingDataPointer->inH;                                \
    (tilingData).inW = tilingDataPointer->inW;                                \
    (tilingData).outD = tilingDataPointer->outD;                              \
    (tilingData).outH = tilingDataPointer->outH;                              \
    (tilingData).outW = tilingDataPointer->outW;                              \
    (tilingData).interpolationMode = tilingDataPointer->interpolationMode;    \
    (tilingData).paddingMode = tilingDataPointer->paddingMode;                \
    (tilingData).alignCorners = tilingDataPointer->alignCorners;              \
    (tilingData).channelLast = tilingDataPointer->channelLast;                \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;                \
    (tilingData).preCoreNum = tilingDataPointer->preCoreNum;                  \
    (tilingData).preNumPerCore = tilingDataPointer->preNumPerCore;            \
    (tilingData).postNumPerCore = tilingDataPointer->postNumPerCore;
#endif