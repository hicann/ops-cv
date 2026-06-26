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
 * \file resize_upsample_trilinear_tiling_data.h
 * \brief ResizeUpsampleTrilinear tiling data for arch35.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_REGBASE_TILING_DATA_H_
#define RESIZE_UPSAMPLE_TRILINEAR_REGBASE_TILING_DATA_H_

#include <cstdint>

struct ResizeUpsampleTrilinearRegBaseTilingData {
    int64_t blkProcessNum;
    int64_t inD;
    int64_t inH;
    int64_t inW;
    int64_t outD;
    int64_t outH;
    int64_t outW;
    int32_t tailBlockNum;
    int32_t alignCorners;
    float scaleD;
    float scaleH;
    float scaleW;
};

#endif // RESIZE_UPSAMPLE_TRILINEAR_REGBASE_TILING_DATA_H_
