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
 * \file deformable_roi_pool_tiling_data.h
 * \brief Tiling data struct for deformable_roi_pool operator
 */

#ifndef DEFORMABLE_ROI_POOL_TILING_DATA_H_
#define DEFORMABLE_ROI_POOL_TILING_DATA_H_

struct DeformableRoiPoolTilingData {
    int32_t needCoreNum = 0;
    int32_t perCoreRois = 0;
    int32_t numRois = 0;
    int32_t N = 0;
    int32_t C = 0;
    int32_t H = 0;
    int32_t W = 0;
    int32_t pooledHeight = 0;
    int32_t pooledWidth = 0;
    int32_t samplingRatio = 0;
    float spatialScale = 0.0f;
    float gamma = 0.0f;
};

#endif // DEFORMABLE_ROI_POOL_TILING_DATA_H_
