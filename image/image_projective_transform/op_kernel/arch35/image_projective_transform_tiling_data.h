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
 * \file image_projective_transform_tiling_data.h
 * \brief Tiling data struct for image_projective_transform operator
 */

#ifndef IMAGE_PROJECTIVE_TRANSFORM_TILING_DATA_H_
#define IMAGE_PROJECTIVE_TRANSFORM_TILING_DATA_H_

struct ImageProjectiveTransformTilingData {
    int32_t needCoreNum = 0;
    int64_t totalPixels = 0;
    int32_t batchSize = 0;
    int32_t hIn = 0;
    int32_t wIn = 0;
    int32_t hOut = 0;
    int32_t wOut = 0;
    int32_t channels = 0;
    int64_t spatialSize = 0;
};

#endif // IMAGE_PROJECTIVE_TRANSFORM_TILING_DATA_H_
