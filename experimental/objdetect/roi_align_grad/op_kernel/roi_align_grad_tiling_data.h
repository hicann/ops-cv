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
 * \file roi_align_grad_tiling_data.h
 * \brief RoiAlignGrad tiling data definition.
 */

#ifndef __ROI_ALIGN_GRAD_TILING_DATA_H__
#define __ROI_ALIGN_GRAD_TILING_DATA_H__

#include <cstdint>

struct RoiAlignGradTilingData {
    uint64_t tilingKey = 0;
    uint64_t runningCoreNum = 1;

    uint64_t roiCount = 0;
    uint64_t roisRowSize = 0;

    uint64_t xDiffN = 0;
    uint64_t xDiffC = 0;
    uint64_t c1 = 0;
    uint64_t xDiffH = 0;
    uint64_t xDiffW = 0;
    uint64_t c1BatchMax = 1;

    int32_t pooledWidth = 0;
    int32_t pooledHeight = 0;
    int32_t sampleNum = 0;
    int32_t roiEndMode = 0;
    int32_t isNd = 0;

    float spatialScale = 0.0F;
    float pooledWidthReciprocal = 0.0F;
    float pooledHeightReciprocal = 0.0F;
    float sampleNumReciprocal = 0.0F;
};

struct RoiAlignGradCompileInfo {
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
};

#endif // __ROI_ALIGN_GRAD_TILING_DATA_H__
