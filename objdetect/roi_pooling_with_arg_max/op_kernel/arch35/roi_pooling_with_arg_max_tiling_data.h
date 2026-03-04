/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max_tiling_data.h
 * \brief roi_pooling_with_arg_max tiling data struct
 */

#ifndef _ROI_POOLING_WITH_ARG_MAX_REGBASE_TILING_DATA_H_
#define _ROI_POOLING_WITH_ARG_MAX_REGBASE_TILING_DATA_H_

#include <string>

struct RoiPoolingWithArgMaxRegBaseTilingData {
    int64_t channels;
    int64_t fmHeight;
    int64_t fmWidth;
    int64_t roiNumber;
    int64_t poolH;
    int64_t poolW;
    float spatialH;
    float spatialW;

    std::string toString() const {
        return "channels = " + std::to_string(channels) + ", fmHeight = " + std::to_string(fmHeight) +
               ", fmWidth = " + std::to_string(fmWidth) + ", roiNumber = " + std::to_string(roiNumber) +
               ", poolH = " + std::to_string(poolH) + ", poolW = " + std::to_string(poolW) +
               ", spatialH = " + std::to_string(spatialH) + ", spatialW = " + std::to_string(spatialW);
    }
};
#endif
